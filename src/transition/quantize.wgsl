// GPU-side color quantization: maps RGBA pixels to palette indices
// using a uniform 6x7x6 RGB cube (252 colours + 1 transparent = 253).
//
// Each thread processes 4 consecutive pixels and packs their
// indices into a single u32 (one byte per index, little-endian).

struct QuantizeParams {
    pixel_count: u32,
    width: u32,
    index_offset: u32,
    _pad: u32,
};

@group(0) @binding(0)
var render_tex: texture_2d<f32>;

@group(0) @binding(1)
var<storage, read_write> index_words: array<u32>;

@group(0) @binding(2)
var<uniform> params: QuantizeParams;

const R_LEVELS: f32 = 6.0;
const G_LEVELS: f32 = 7.0;
const B_LEVELS: f32 = 6.0;
const TRANSPARENT_INDEX: u32 = 252u;

fn quantize_one_vec(color: vec4<f32>) -> u32 {
    if (color.a < 0.5) {
        return TRANSPARENT_INDEX;
    }
    let ri = u32(clamp(round(color.r * (R_LEVELS - 1.0)), 0.0, R_LEVELS - 1.0));
    let gi = u32(clamp(round(color.g * (G_LEVELS - 1.0)), 0.0, G_LEVELS - 1.0));
    let bi = u32(clamp(round(color.b * (B_LEVELS - 1.0)), 0.0, B_LEVELS - 1.0));
    return ri * u32(G_LEVELS) * u32(B_LEVELS) + gi * u32(B_LEVELS) + bi;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words_per_frame = (params.pixel_count + 3u) / 4u;
    if (gid.x >= words_per_frame) {
        return;
    }

    let base_pixel = gid.x * 4u;
    let remaining = params.pixel_count - base_pixel;
    let width = params.width;

    var out: u32 = 0u;
    let p0 = base_pixel;
    out = quantize_one_vec(textureLoad(render_tex, vec2<u32>(p0 % width, p0 / width), 0));
    if (remaining > 1u) {
        let p1 = base_pixel + 1u;
        out = out | (quantize_one_vec(textureLoad(render_tex, vec2<u32>(p1 % width, p1 / width), 0)) << 8u);
    }
    if (remaining > 2u) {
        let p2 = base_pixel + 2u;
        out = out | (quantize_one_vec(textureLoad(render_tex, vec2<u32>(p2 % width, p2 / width), 0)) << 16u);
    }
    if (remaining > 3u) {
        let p3 = base_pixel + 3u;
        out = out | (quantize_one_vec(textureLoad(render_tex, vec2<u32>(p3 % width, p3 / width), 0)) << 24u);
    }

    index_words[params.index_offset + gid.x] = out;
}
