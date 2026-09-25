"""Headless OpenGL renderer (moderngl, EGL): points, lines and a glass shell drawn
additively into a float HDR buffer, then bloom, ACES tone mapping and grain.

Every light source is additive, so a frame is the sum of its layers; temporal
accumulation (motion blur over sub-frames) is done by drawing each sub-frame
with weight 1/K into the same float accumulation buffer.

On a machine with a GPU this uses it; here it runs on Mesa llvmpipe (CPU). See
render/cpu.py for the NumPy fallback used when no GL context is available.
"""
from __future__ import annotations

import numpy as np

try:
    import moderngl
except ImportError:  # pragma: no cover - CPU fallback handles this
    moderngl = None

POINT_VS = """
#version 330
uniform mat4 u_mvp;
uniform float u_px;          // pixels per unit of (world size / clip w)
uniform float u_min_px;      // smallest sprite drawn (flux is conserved below it)
uniform float u_max_px;
uniform float u_gain;        // global intensity multiplier (sub-frame weight etc.)
in vec3 in_pos;
in vec3 in_color;            // linear RGB, already multiplied by intensity
in float in_size;            // world-space diameter
out vec3 v_color;
void main() {
    vec4 clip = u_mvp * vec4(in_pos, 1.0);
    gl_Position = clip;
    float px = in_size * u_px / max(clip.w, 1e-4);
    float drawn = clamp(px, u_min_px, u_max_px);
    float flux = (px * px) / (drawn * drawn);     // keep total light when clamped up
    v_color = in_color * u_gain * min(flux, 1.0);
    gl_PointSize = drawn;
    if (clip.w <= 0.0 || dot(in_color, in_color) <= 0.0) gl_PointSize = 0.0;
}
"""

POINT_FS = """
#version 330
uniform float u_ring;        // 0 = soft dot, 1 = thin ring marker
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float a;
    if (u_ring > 0.5) {
        float r = sqrt(r2);
        a = exp(-pow((r - 0.78) / 0.07, 2.0));
    } else {
        a = exp(-4.5 * r2);
    }
    f_color = vec4(v_color * a, 1.0);
}
"""

LINE_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_color;
out vec3 g_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    g_color = in_color;
}
"""

LINE_GS = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 u_viewport;     // pixels
uniform float u_width_px;    // full width of the soft line
uniform float u_gain;
in vec3 g_color[];
out vec3 f_col;
out float f_across;
void main() {
    vec4 p0 = gl_in[0].gl_Position, p1 = gl_in[1].gl_Position;
    if (p0.w <= 0.0 || p1.w <= 0.0) return;
    vec2 s0 = p0.xy / p0.w * u_viewport * 0.5;
    vec2 s1 = p1.xy / p1.w * u_viewport * 0.5;
    vec2 dir = s1 - s0;
    float len = length(dir);
    if (len < 1e-4) dir = vec2(1.0, 0.0); else dir /= len;
    vec2 nrm = vec2(-dir.y, dir.x) * (u_width_px * 0.5);
    vec2 o0 = nrm / (u_viewport * 0.5) * p0.w;
    vec2 o1 = nrm / (u_viewport * 0.5) * p1.w;
    // lines thinner than ~1.5 px are drawn at 1.5 px with proportionally less light
    float k = u_gain;
    f_col = g_color[0] * k; f_across = -1.0; gl_Position = vec4(p0.xy - o0, p0.zw); EmitVertex();
    f_col = g_color[0] * k; f_across =  1.0; gl_Position = vec4(p0.xy + o0, p0.zw); EmitVertex();
    f_col = g_color[1] * k; f_across = -1.0; gl_Position = vec4(p1.xy - o1, p1.zw); EmitVertex();
    f_col = g_color[1] * k; f_across =  1.0; gl_Position = vec4(p1.xy + o1, p1.zw); EmitVertex();
    EndPrimitive();
}
"""

LINE_FS = """
#version 330
in vec3 f_col;
in float f_across;
out vec4 f_color;
void main() {
    float a = exp(-3.0 * f_across * f_across);
    f_color = vec4(f_col * a, 1.0);
}
"""

SHELL_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_pos;
out vec3 v_nrm;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_pos = in_pos;
    v_nrm = in_normal;
}
"""

SHELL_FS = """
#version 330
uniform vec3 u_eye;
uniform vec3 u_color;
uniform float u_opacity;
uniform float u_power;
uniform float u_fill;
in vec3 v_pos;
in vec3 v_nrm;
out vec4 f_color;
void main() {
    float ln = length(v_nrm);
    if (ln < 1e-6) discard;
    vec3 n = v_nrm / ln;
    vec3 v = normalize(u_eye - v_pos);
    float facing = clamp(abs(dot(n, v)), 0.0, 1.0);   // rounding can push |n.v| above 1; pow() of a negative base is NaN
    float rim = pow(1.0 - facing, u_power);
    f_color = vec4(u_color * u_opacity * (rim + u_fill), 1.0);
}
"""

FULLSCREEN_VS = """
#version 330
in vec2 in_xy;
out vec2 v_uv;
void main() { v_uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

DOWN_FS = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_texel;        // 1 / source size
in vec2 v_uv;
out vec4 f_color;
vec3 tex(vec2 uv) {  // never let a non-finite value into the bloom chain
    vec3 c = texture(u_src, uv).rgb;
    return (any(isnan(c)) || any(isinf(c))) ? vec3(0.0) : c;
}
void main() {   // 13-tap filter (Jimenez 2014, "Next generation post processing in Call of Duty")
    vec2 t = u_texel;
    vec3 a = tex(v_uv + t * vec2(-2, 2));
    vec3 b = tex(v_uv + t * vec2( 0, 2));
    vec3 c = tex(v_uv + t * vec2( 2, 2));
    vec3 d = tex(v_uv + t * vec2(-2, 0));
    vec3 e = tex(v_uv);
    vec3 f = tex(v_uv + t * vec2( 2, 0));
    vec3 g = tex(v_uv + t * vec2(-2,-2));
    vec3 h = tex(v_uv + t * vec2( 0,-2));
    vec3 i = tex(v_uv + t * vec2( 2,-2));
    vec3 j = tex(v_uv + t * vec2(-1, 1));
    vec3 k = tex(v_uv + t * vec2( 1, 1));
    vec3 l = tex(v_uv + t * vec2(-1,-1));
    vec3 m = tex(v_uv + t * vec2( 1,-1));
    vec3 col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    f_color = vec4(col, 1.0);
}
"""

UP_FS = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_radius;
in vec2 v_uv;
out vec4 f_color;
void main() {   // 9-tap tent filter, added onto the next larger level
    vec2 t = u_texel * u_radius;
    vec3 col = texture(u_src, v_uv).rgb * 4.0;
    col += (texture(u_src, v_uv + vec2(-t.x, 0)).rgb + texture(u_src, v_uv + vec2(t.x, 0)).rgb
          + texture(u_src, v_uv + vec2(0, -t.y)).rgb + texture(u_src, v_uv + vec2(0, t.y)).rgb) * 2.0;
    col += texture(u_src, v_uv + vec2(-t.x, -t.y)).rgb + texture(u_src, v_uv + vec2(t.x, -t.y)).rgb
         + texture(u_src, v_uv + vec2(-t.x, t.y)).rgb + texture(u_src, v_uv + vec2(t.x, t.y)).rgb;
    f_color = vec4(col / 16.0, 1.0);
}
"""

FINAL_FS = """
#version 330
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_strength;
uniform float u_grain;
uniform float u_seed;
uniform vec3 u_bg;           // display-referred background (sRGB 0..1)
uniform float u_fade;        // global fade to black (1 = visible)
in vec2 v_uv;
out vec4 f_color;
vec3 aces(vec3 x) {          // ACES filmic curve, Narkowicz 2015 fit
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main() {
    vec3 base = texture(u_hdr, v_uv).rgb;
    if (any(isnan(base)) || any(isinf(base))) base = vec3(0.0);
    vec3 hdr = base + u_bloom_strength * texture(u_bloom, v_uv).rgb;
    vec3 c = aces(hdr * u_exposure) * u_fade;
    c = pow(c, vec3(1.0 / 2.2));
    c = u_bg + c * (1.0 - u_bg);
    float n = hash(gl_FragCoord.xy + u_seed * 97.13) + hash(gl_FragCoord.yx * 1.37 + u_seed * 13.7) - 1.0;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c += n * u_grain * (0.35 + 0.65 * sqrt(lum));
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


class GLRenderer:
    """Owns the GL context, buffers and programs for one output resolution."""

    def __init__(self, width: int, height: int, bloom_levels: int = 6):
        if moderngl is None:
            raise RuntimeError("moderngl not available")
        try:
            self.ctx = moderngl.create_context(standalone=True, backend="egl")
        except Exception:
            self.ctx = moderngl.create_context(standalone=True)
        ctx = self.ctx
        self.size = (int(width), int(height))
        self.renderer_name = ctx.info.get("GL_RENDERER", "?")
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.accum_tex = ctx.texture(self.size, 4, dtype="f4")
        self.accum = ctx.framebuffer(color_attachments=[self.accum_tex])
        self.levels = []
        w, h = self.size
        for _ in range(bloom_levels):
            w, h = max(1, w // 2), max(1, h // 2)
            tex = ctx.texture((w, h), 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            self.levels.append((tex, ctx.framebuffer(color_attachments=[tex])))
        self.accum_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.accum_tex.repeat_x = self.accum_tex.repeat_y = False
        self.out_tex = ctx.texture(self.size, 4, dtype="f1")
        self.out = ctx.framebuffer(color_attachments=[self.out_tex])

        self.p_points = ctx.program(vertex_shader=POINT_VS, fragment_shader=POINT_FS)
        self.p_lines = ctx.program(vertex_shader=LINE_VS, geometry_shader=LINE_GS, fragment_shader=LINE_FS)
        self.p_shell = ctx.program(vertex_shader=SHELL_VS, fragment_shader=SHELL_FS)
        self.p_down = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=DOWN_FS)
        self.p_up = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=UP_FS)
        self.p_final = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=FINAL_FS)
        quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
        self.fs = {p: ctx.vertex_array(prog, [(quad, "2f", "in_xy")])
                   for p, prog in (("down", self.p_down), ("up", self.p_up), ("final", self.p_final))}
        self._point_sets: dict = {}
        self._line_sets: dict = {}
        self._shell = None

    # ------------------------------------------------------------ geometry
    def set_shell(self, vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> None:
        vbo = self.ctx.buffer(np.hstack([vertices, normals]).astype("f4").tobytes())
        ibo = self.ctx.buffer(faces.astype("i4").tobytes())
        self._shell = self.ctx.vertex_array(self.p_shell, [(vbo, "3f 3f", "in_pos", "in_normal")], ibo)

    def point_set(self, key: str, positions: np.ndarray):
        """Static positions; colours and sizes are streamed per draw."""
        n = len(positions)
        ps = self._point_sets.get(key)
        if ps is None or ps["n"] != n:
            pos = self.ctx.buffer(positions.astype("f4").tobytes())
            col = self.ctx.buffer(reserve=n * 12, dynamic=True)
            siz = self.ctx.buffer(reserve=n * 4, dynamic=True)
            vao = self.ctx.vertex_array(self.p_points, [(pos, "3f", "in_pos"), (col, "3f", "in_color"),
                                                        (siz, "1f", "in_size")])
            ps = dict(n=n, pos=pos, col=col, siz=siz, vao=vao)
            self._point_sets[key] = ps
        return ps

    def line_set(self, key: str, segments: np.ndarray):
        """segments: (M, 2, 3) static endpoints; colours streamed per draw."""
        m = len(segments)
        ls = self._line_sets.get(key)
        if ls is None or ls["m"] != m:
            pos = self.ctx.buffer(segments.reshape(-1, 3).astype("f4").tobytes())
            col = self.ctx.buffer(reserve=max(1, m * 2 * 12), dynamic=True)
            vao = self.ctx.vertex_array(self.p_lines, [(pos, "3f", "in_pos"), (col, "3f", "in_color")])
            ls = dict(m=m, pos=pos, col=col, vao=vao)
            self._line_sets[key] = ls
        return ls

    # ------------------------------------------------------------- drawing
    def begin(self) -> None:
        self.accum.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.accum.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def _common(self, prog, cam):
        prog["u_mvp"].write(cam["mvp"].T.astype("f4").tobytes())

    def draw_shell(self, cam, color, opacity, power=3.0, fill=0.0) -> None:
        if self._shell is None or opacity <= 0:
            return
        p = self.p_shell
        self._common(p, cam)
        p["u_eye"].value = tuple(float(x) for x in cam["eye"])
        p["u_color"].value = tuple(float(x) for x in color)
        p["u_opacity"].value = float(opacity)
        p["u_power"].value = float(power)
        p["u_fill"].value = float(fill)
        self.accum.use()
        self._shell.render(moderngl.TRIANGLES)

    def draw_points(self, key, positions, colors, sizes, cam, gain=1.0, min_px=1.2, max_px=400.0, ring=False):
        ps = self.point_set(key, positions)
        ps["col"].write(np.ascontiguousarray(colors, dtype="f4").tobytes())
        ps["siz"].write(np.ascontiguousarray(sizes, dtype="f4").tobytes())
        p = self.p_points
        self._common(p, cam)
        p["u_px"].value = float(cam["px_per_unit"])
        p["u_min_px"].value = float(min_px)
        p["u_max_px"].value = float(max_px)
        p["u_gain"].value = float(gain)
        p["u_ring"].value = 1.0 if ring else 0.0
        self.accum.use()
        ps["vao"].render(moderngl.POINTS)

    def draw_lines(self, key, segments, colors, cam, width_px=1.5, gain=1.0):
        if len(segments) == 0:
            return
        ls = self.line_set(key, segments)
        ls["col"].write(np.ascontiguousarray(colors, dtype="f4").reshape(-1, 3).tobytes())
        p = self.p_lines
        self._common(p, cam)
        p["u_viewport"].value = tuple(float(x) for x in self.size)
        p["u_width_px"].value = float(width_px)
        p["u_gain"].value = float(gain)
        self.accum.use()
        ls["vao"].render(moderngl.LINES)

    # --------------------------------------------------------- post process
    def finish(self, exposure=1.0, bloom_strength=0.6, bloom_radius=1.0, grain=0.012, seed=0.0,
               bg=(3 / 255, 4 / 255, 7 / 255), fade=1.0) -> np.ndarray:
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        src, src_size = self.accum_tex, self.size
        for tex, fbo in self.levels:            # downsample chain
            fbo.use()
            ctx.viewport = (0, 0, *tex.size)
            src.use(location=0)
            self.p_down["u_src"].value = 0
            self.p_down["u_texel"].value = (1.0 / src_size[0], 1.0 / src_size[1])
            self.fs["down"].render()
            src, src_size = tex, tex.size
        ctx.enable(moderngl.BLEND)              # upsample and add back up the chain
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for (tex_small, _), (tex_big, fbo_big) in zip(self.levels[::-1][:-1], self.levels[::-1][1:]):
            fbo_big.use()
            ctx.viewport = (0, 0, *tex_big.size)
            tex_small.use(location=0)
            self.p_up["u_src"].value = 0
            self.p_up["u_texel"].value = (1.0 / tex_small.size[0], 1.0 / tex_small.size[1])
            self.p_up["u_radius"].value = float(bloom_radius)
            self.fs["up"].render()
        ctx.disable(moderngl.BLEND)
        self.out.use()
        ctx.viewport = (0, 0, *self.size)
        self.accum_tex.use(location=0)
        self.levels[0][0].use(location=1)
        f = self.p_final
        f["u_hdr"].value = 0
        f["u_bloom"].value = 1
        f["u_exposure"].value = float(exposure)
        f["u_bloom_strength"].value = float(bloom_strength) / len(self.levels)
        f["u_grain"].value = float(grain)
        f["u_seed"].value = float(seed % 1000)
        f["u_bg"].value = tuple(float(x) for x in bg)
        f["u_fade"].value = float(fade)
        self.fs["final"].render()
        data = self.out.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)
        return img[::-1]

    def release(self) -> None:
        self.ctx.release()
