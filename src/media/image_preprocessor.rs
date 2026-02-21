use std::ffi::OsStr;
use std::io::Cursor;
use std::path::Path;

use image::DynamicImage;
use resvg::{tiny_skia, usvg};
use thiserror::Error;

use super::PanelDimensions;

/// Errors returned while decoding and normalising input images.
#[derive(Debug, Error)]
pub enum ImagePreparationError {
    /// The source file cannot be read.
    #[error("failed to read source image")]
    Read(#[source] std::io::Error),
    /// The source bytes are not a supported image format.
    #[error("failed to detect image format from source bytes")]
    UnknownFormat(#[source] image::ImageError),
    /// The source image failed to decode.
    #[error("failed to decode source image")]
    Decode(#[source] image::ImageError),
    /// SVG parsing failed.
    #[error("failed to parse SVG source")]
    SvgParse(#[source] usvg::Error),
    /// SVG raster surface allocation failed.
    #[error("failed to allocate SVG raster surface ({width}x{height})")]
    SvgPixmapAlloc {
        /// Raster target width.
        width: u32,
        /// Raster target height.
        height: u32,
    },
    /// SVG raster output buffer does not match the expected image dimensions.
    #[error("SVG raster buffer size mismatch ({width}x{height})")]
    SvgBufferMismatch {
        /// Expected image width.
        width: u32,
        /// Expected image height.
        height: u32,
    },
}

/// Returns `true` when `path` has an SVG extension or `bytes` look like SVG
/// markup.
pub(crate) fn is_svg(path: &Path, bytes: &[u8]) -> bool {
    has_svg_extension(path) || looks_like_svg(bytes)
}

/// Decodes a raster image from raw bytes, applying EXIF orientation.
pub(crate) fn decode_raster(bytes: &[u8]) -> Result<image::RgbaImage, ImagePreparationError> {
    let format = image::guess_format(bytes).map_err(ImagePreparationError::UnknownFormat)?;
    let decoded = image::load_from_memory_with_format(bytes, format)
        .map_err(ImagePreparationError::Decode)?;
    let oriented = apply_orientation(decoded, exif_orientation(bytes));
    Ok(oriented.to_rgba8())
}

/// Decodes an SVG image from raw bytes, optionally rasterising at a target
/// size.
pub(crate) fn decode_svg(
    bytes: &[u8],
    fit_to: Option<PanelDimensions>,
) -> Result<image::RgbaImage, ImagePreparationError> {
    let options = usvg::Options::default();
    let tree =
        usvg::Tree::from_data(bytes, &options).map_err(ImagePreparationError::SvgParse)?;
    let svg_size = tree.size();

    let (width, height, transform) = match fit_to {
        Some(target) => {
            let scale = f32::min(
                f32::from(target.width()) / svg_size.width(),
                f32::from(target.height()) / svg_size.height(),
            );
            let w = (svg_size.width() * scale).round().max(1.0) as u32;
            let h = (svg_size.height() * scale).round().max(1.0) as u32;
            (w, h, tiny_skia::Transform::from_scale(scale, scale))
        }
        None => {
            let int_size = svg_size.to_int_size();
            (
                int_size.width(),
                int_size.height(),
                tiny_skia::Transform::default(),
            )
        }
    };

    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or(ImagePreparationError::SvgPixmapAlloc {
            width,
            height,
        })?;
    let mut pixmap_mut = pixmap.as_mut();
    resvg::render(&tree, transform, &mut pixmap_mut);

    let rgba = pixmap.take_demultiplied();
    image::RgbaImage::from_raw(width, height, rgba)
        .ok_or(ImagePreparationError::SvgBufferMismatch { width, height })
}

fn has_svg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("svg") || extension.eq_ignore_ascii_case("svgz")
        })
}

fn looks_like_svg(source_bytes: &[u8]) -> bool {
    let sniff_end = source_bytes.len().min(2_048);
    let sniff = &source_bytes[..sniff_end];
    let text = match std::str::from_utf8(sniff) {
        Ok(text) => text,
        Err(_) => return false,
    };
    let trimmed = text.trim_start_matches(|character: char| {
        character == '\u{FEFF}' || character.is_ascii_whitespace()
    });
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("<svg") || (lower.starts_with("<?xml") && lower.contains("<svg"))
}

fn apply_orientation(image: DynamicImage, orientation: Option<u32>) -> DynamicImage {
    match orientation {
        Some(2) => image.fliph(),
        Some(3) => image.rotate180(),
        Some(4) => image.flipv(),
        Some(5) => image.fliph().rotate90(),
        Some(6) => image.rotate90(),
        Some(7) => image.fliph().rotate270(),
        Some(8) => image.rotate270(),
        _ => image,
    }
}

fn exif_orientation(source_bytes: &[u8]) -> Option<u32> {
    let mut cursor = Cursor::new(source_bytes);
    let exif = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?
        .value
        .get_uint(0)
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use image::ImageEncoder;
    use pretty_assertions::assert_eq;
    use svg::Document;
    use svg::node::element::Rectangle;

    use super::*;

    fn svg_filled_rect(width: u32, height: u32, fill: &str) -> String {
        Document::new()
            .set("xmlns", "http://www.w3.org/2000/svg")
            .set("width", width)
            .set("height", height)
            .add(
                Rectangle::new()
                    .set("width", width)
                    .set("height", height)
                    .set("fill", fill),
            )
            .to_string()
    }

    #[test]
    fn decode_raster_preserves_dimensions() -> Result<(), Box<dyn std::error::Error>> {
        let mut png_bytes = Vec::new();
        let source = image::RgbaImage::from_pixel(3, 2, image::Rgba([0xAA, 0xBB, 0xCC, 0xFF]));
        image::codecs::png::PngEncoder::new(&mut png_bytes).write_image(
            source.as_raw(),
            3,
            2,
            image::ExtendedColorType::Rgba8,
        )?;

        let decoded = decode_raster(&png_bytes)?;

        assert_eq!(3, decoded.width());
        assert_eq!(2, decoded.height());
        Ok(())
    }

    #[test]
    fn detects_svg_from_markup() {
        let svg = svg_filled_rect(1, 1, "black");

        assert!(looks_like_svg(svg.as_bytes()));
    }

    #[test]
    fn rasterises_svg() -> Result<(), Box<dyn std::error::Error>> {
        let svg = svg_filled_rect(3, 2, "#112233");

        let decoded = decode_svg(svg.as_bytes(), None)?;

        assert_eq!(3, decoded.width());
        assert_eq!(2, decoded.height());
        assert_eq!(
            image::Rgba([0x11, 0x22, 0x33, 0xFF]),
            *decoded.get_pixel(1, 1)
        );
        Ok(())
    }

    #[test]
    fn rasterises_svg_at_target_size() -> Result<(), Box<dyn std::error::Error>> {
        let svg = svg_filled_rect(10, 5, "#AABBCC");
        let target = PanelDimensions::new(100, 100).expect("100x100 should be valid");

        let decoded = decode_svg(svg.as_bytes(), Some(target))?;

        // 10x5 SVG fitted into 100x100: scale = min(10, 20) = 10.
        // Rasterised at 100x50.
        assert_eq!(100, decoded.width());
        assert_eq!(50, decoded.height());
        assert_eq!(
            image::Rgba([0xAA, 0xBB, 0xCC, 0xFF]),
            *decoded.get_pixel(50, 25)
        );
        Ok(())
    }

    #[test]
    fn unknown_bytes_report_unknown_format() {
        let bytes = [0x00, 0x01, 0x02, 0x03];

        assert_matches!(decode_raster(&bytes), Err(ImagePreparationError::UnknownFormat(_)));
    }
}
