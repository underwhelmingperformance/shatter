use std::ffi::OsStr;
use std::io::Cursor;
use std::path::Path;

use image::DynamicImage;
use resvg::{tiny_skia, usvg};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum InputKind {
    Raster(image::ImageFormat),
    Svg,
}

impl InputKind {
    fn detect(path: &Path, source_bytes: &[u8]) -> Result<Self, ImagePreparationError> {
        if has_svg_extension(path) || looks_like_svg(source_bytes) {
            return Ok(Self::Svg);
        }

        let source_format =
            image::guess_format(source_bytes).map_err(ImagePreparationError::UnknownFormat)?;
        Ok(Self::Raster(source_format))
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Raster(_) => "raster",
            Self::Svg => "svg",
        }
    }
}

#[derive(Debug)]
struct ImageSource<'a> {
    bytes: &'a [u8],
    kind: InputKind,
}

impl<'a> ImageSource<'a> {
    fn from_parts(path: &'a Path, bytes: &'a [u8]) -> Result<Self, ImagePreparationError> {
        Ok(Self {
            bytes,
            kind: InputKind::detect(path, bytes)?,
        })
    }

    const fn kind(&self) -> InputKind {
        self.kind
    }

    const fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

trait ImageDecodeBackend {
    fn decode(&self, source: &ImageSource<'_>) -> Result<image::RgbaImage, ImagePreparationError>;
}

#[derive(Debug, Default)]
struct RasterImageDecodeBackend;

impl ImageDecodeBackend for RasterImageDecodeBackend {
    fn decode(&self, source: &ImageSource<'_>) -> Result<image::RgbaImage, ImagePreparationError> {
        let source_format = match source.kind() {
            InputKind::Raster(source_format) => source_format,
            other => {
                return Err(ImagePreparationError::BackendRouting {
                    expected: "raster",
                    actual: other.as_str(),
                });
            }
        };

        let decoded = image::load_from_memory_with_format(source.bytes(), source_format)
            .map_err(ImagePreparationError::Decode)?;
        let oriented = apply_orientation(decoded, exif_orientation(source.bytes()));
        Ok(oriented.to_rgba8())
    }
}

#[derive(Debug, Default)]
struct SvgImageDecodeBackend;

impl ImageDecodeBackend for SvgImageDecodeBackend {
    fn decode(&self, source: &ImageSource<'_>) -> Result<image::RgbaImage, ImagePreparationError> {
        if !matches!(source.kind(), InputKind::Svg) {
            return Err(ImagePreparationError::BackendRouting {
                expected: "svg",
                actual: source.kind().as_str(),
            });
        }

        let options = usvg::Options::default();
        let tree = usvg::Tree::from_data(source.bytes(), &options)
            .map_err(ImagePreparationError::SvgParse)?;
        let target_size = tree.size().to_int_size();
        let width = target_size.width();
        let height = target_size.height();

        let mut pixmap = tiny_skia::Pixmap::new(width, height).ok_or({
            ImagePreparationError::SvgRasterise {
                width,
                height,
                reason: "failed to allocate SVG raster surface",
            }
        })?;
        let mut pixmap_mut = pixmap.as_mut();
        resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap_mut);

        let rgba = pixmap.take_demultiplied();
        image::RgbaImage::from_raw(width, height, rgba).ok_or(ImagePreparationError::SvgRasterise {
            width,
            height,
            reason: "SVG raster output has an unexpected pixel buffer size",
        })
    }
}

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
    /// Backend routing received an unexpected source format.
    #[error("failed to route image decode backend: expected {expected}, got {actual}")]
    BackendRouting {
        /// Decoder backend expected input kind.
        expected: &'static str,
        /// Actual input kind.
        actual: &'static str,
    },
    /// SVG parsing failed.
    #[error("failed to parse SVG source")]
    SvgParse(#[source] usvg::Error),
    /// SVG rasterisation failed.
    #[error("failed to rasterise SVG source ({width}x{height}): {reason}")]
    SvgRasterise {
        /// Raster target width.
        width: u32,
        /// Raster target height.
        height: u32,
        /// Failure reason.
        reason: &'static str,
    },
}

#[derive(Debug, Default)]
pub(crate) struct ImagePreprocessor {
    raster_backend: RasterImageDecodeBackend,
    svg_backend: SvgImageDecodeBackend,
}

impl ImagePreprocessor {
    pub(crate) fn decode_oriented_from_path(
        path: &Path,
    ) -> Result<image::RgbaImage, ImagePreparationError> {
        Self::default().decode_from_path(path)
    }

    fn decode_from_path(&self, path: &Path) -> Result<image::RgbaImage, ImagePreparationError> {
        let source_bytes = std::fs::read(path).map_err(ImagePreparationError::Read)?;
        let source = ImageSource::from_parts(path, &source_bytes)?;
        self.decode_source(&source)
    }

    fn decode_source(
        &self,
        source: &ImageSource<'_>,
    ) -> Result<image::RgbaImage, ImagePreparationError> {
        match source.kind() {
            InputKind::Raster(_) => self.raster_backend.decode(source),
            InputKind::Svg => self.svg_backend.decode(source),
        }
    }
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
    use std::path::Path;

    use assert_matches::assert_matches;
    use image::ImageEncoder;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn decode_oriented_from_bytes_preserves_dimensions_when_no_exif_rotation()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut png_bytes = Vec::new();
        let source = image::RgbaImage::from_pixel(3, 2, image::Rgba([0xAA, 0xBB, 0xCC, 0xFF]));
        image::codecs::png::PngEncoder::new(&mut png_bytes).write_image(
            source.as_raw(),
            3,
            2,
            image::ExtendedColorType::Rgba8,
        )?;
        let image_source = ImageSource::from_parts(Path::new("test.png"), &png_bytes)?;
        let preprocessor = ImagePreprocessor::default();

        let decoded = preprocessor.decode_source(&image_source)?;

        assert_eq!(3, decoded.width());
        assert_eq!(2, decoded.height());
        Ok(())
    }

    #[test]
    fn detects_svg_sources_from_extension_or_markup() -> Result<(), Box<dyn std::error::Error>> {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>"#;
        let source_by_extension = ImageSource::from_parts(Path::new("icon.svg"), svg)?;
        let source_by_markup = ImageSource::from_parts(Path::new("icon"), svg)?;

        assert_eq!(InputKind::Svg, source_by_extension.kind());
        assert_eq!(InputKind::Svg, source_by_markup.kind());
        Ok(())
    }

    #[test]
    fn rasterises_svg_sources() -> Result<(), Box<dyn std::error::Error>> {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="3" height="2"><rect width="3" height="2" fill="#112233"/></svg>"##;
        let source = ImageSource::from_parts(Path::new("panel.svg"), svg)?;
        let preprocessor = ImagePreprocessor::default();

        let decoded = preprocessor.decode_source(&source)?;

        assert_eq!(3, decoded.width());
        assert_eq!(2, decoded.height());
        assert_eq!(
            image::Rgba([0x11, 0x22, 0x33, 0xFF]),
            *decoded.get_pixel(1, 1)
        );
        Ok(())
    }

    #[test]
    fn unknown_binary_sources_report_unknown_format() {
        let bytes = [0x00, 0x01, 0x02, 0x03];
        let result = ImageSource::from_parts(Path::new("blob.bin"), &bytes);

        assert_matches!(result, Err(ImagePreparationError::UnknownFormat(_)));
    }
}
