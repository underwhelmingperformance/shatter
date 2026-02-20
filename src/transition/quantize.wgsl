// GPU-side color quantization: maps RGBA pixels to palette indices
// using a uniform 6x7x6 RGB cube (252 colours + 1 transparent = 253).
//
// The 6x7x6 cube gives 252 entries which, together with one
// transparent slot, fits comfortably in a 256-colour GIF palette
// while providing slightly better green resolution (the human eye
// is most sensitive to green).
//
// Each thread processes 4 consecutive pixels and packs their
// indices into a single u32 (one byte per index, little-endian).

struct QuantizeParams {
    pixel_count: u32,
    chunk_frames: u32,
};

@group(0) @binding(0)
var<storage, read> rgba_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> index_words: array<u32>;

@group(0) @binding(2)
var<uniform> params: QuantizeParams;

const R_LEVELS: f32 = 6.0;
const G_LEVELS: f32 = 7.0;
const B_LEVELS: f32 = 6.0;
const TRANSPARENT_INDEX: u32 = 252u;

fn quantize_one(packed: u32) -> u32 {
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    if (a < 0.5) {
        return TRANSPARENT_INDEX;
    }
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;

    let ri = u32(clamp(round(r * (R_LEVELS - 1.0)), 0.0, R_LEVELS - 1.0));
    let gi = u32(clamp(round(g * (G_LEVELS - 1.0)), 0.0, G_LEVELS - 1.0));
    let bi = u32(clamp(round(b * (B_LEVELS - 1.0)), 0.0, B_LEVELS - 1.0));
    return ri * u32(G_LEVELS) * u32(B_LEVELS) + gi * u32(B_LEVELS) + bi;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread packs 4 pixels into one output u32.
    let words_per_frame = (params.pixel_count + 3u) / 4u;
    let word_id = gid.z * words_per_frame + gid.x;
    if (gid.x >= words_per_frame) {
        return;
    }

    let base_pixel = gid.z * params.pixel_count + gid.x * 4u;
    let remaining = params.pixel_count - gid.x * 4u;

    var out: u32 = 0u;
    // Always at least 1 pixel (gid.x < words_per_frame guarantees it).
    out = quantize_one(rgba_pixels[base_pixel]);
    if (remaining > 1u) {
        out = out | (quantize_one(rgba_pixels[base_pixel + 1u]) << 8u);
    }
    if (remaining > 2u) {
        out = out | (quantize_one(rgba_pixels[base_pixel + 2u]) << 16u);
    }
    if (remaining > 3u) {
        out = out | (quantize_one(rgba_pixels[base_pixel + 3u]) << 24u);
    }

    index_words[word_id] = out;
}
