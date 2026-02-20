use std::fmt::{self, Display, Formatter};
use std::num::ParseIntError;
use std::str::FromStr;

use thiserror::Error;

/// Concrete output dimensions in pixels.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PanelDimensions {
    width: u16,
    height: u16,
}

impl PanelDimensions {
    /// Creates dimensions when both values are non-zero.
    ///
    /// ```
    /// use shatter::PanelDimensions;
    ///
    /// let dimensions = PanelDimensions::new(64, 64).expect("64x64 should be valid");
    /// assert_eq!(64, dimensions.width());
    /// assert_eq!(64, dimensions.height());
    /// ```
    #[must_use]
    pub const fn new(width: u16, height: u16) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }

        Some(Self { width, height })
    }

    /// Returns width in pixels.
    ///
    /// ```
    /// use shatter::PanelDimensions;
    ///
    /// let dimensions = PanelDimensions::new(8, 32).expect("8x32 should be valid");
    /// assert_eq!(8, dimensions.width());
    /// ```
    #[must_use]
    pub const fn width(self) -> u16 {
        self.width
    }

    /// Returns height in pixels.
    ///
    /// ```
    /// use shatter::PanelDimensions;
    ///
    /// let dimensions = PanelDimensions::new(8, 32).expect("8x32 should be valid");
    /// assert_eq!(32, dimensions.height());
    /// ```
    #[must_use]
    pub const fn height(self) -> u16 {
        self.height
    }

    /// Returns total pixel count.
    ///
    /// ```
    /// use shatter::PanelDimensions;
    ///
    /// let dimensions = PanelDimensions::new(8, 32).expect("8x32 should be valid");
    /// assert_eq!(256, dimensions.area());
    /// ```
    #[must_use]
    pub const fn area(self) -> u32 {
        (self.width as u32) * (self.height as u32)
    }
}

impl Display for PanelDimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl FromStr for PanelDimensions {
    type Err = PanelDimensionsParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (width_raw, height_raw) = value
            .split_once(['x', 'X'])
            .ok_or(PanelDimensionsParseError::InvalidFormat)?;
        let width =
            width_raw
                .parse::<u16>()
                .map_err(|source| PanelDimensionsParseError::InvalidWidth {
                    value: width_raw.to_string(),
                    source,
                })?;
        let height = height_raw.parse::<u16>().map_err(|source| {
            PanelDimensionsParseError::InvalidHeight {
                value: height_raw.to_string(),
                source,
            }
        })?;

        PanelDimensions::new(width, height).ok_or(PanelDimensionsParseError::ZeroDimension)
    }
}

/// Errors returned when parsing `WIDTHxHEIGHT` values.
#[derive(Debug, Error)]
pub enum PanelDimensionsParseError {
    /// The value is not in `WIDTHxHEIGHT` format.
    #[error("dimensions must use WIDTHxHEIGHT format")]
    InvalidFormat,
    /// Width cannot be parsed to an unsigned integer.
    #[error("invalid width value `{value}`")]
    InvalidWidth {
        /// Original width text.
        value: String,
        /// Parse failure source.
        source: ParseIntError,
    },
    /// Height cannot be parsed to an unsigned integer.
    #[error("invalid height value `{value}`")]
    InvalidHeight {
        /// Original height text.
        value: String,
        /// Parse failure source.
        source: ParseIntError,
    },
    /// Width or height is zero.
    #[error("dimensions must be non-zero")]
    ZeroDimension,
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("64x64", PanelDimensions::new(64, 64).expect("64x64 should be valid"))]
    #[case("8X32", PanelDimensions::new(8, 32).expect("8x32 should be valid"))]
    fn parse_accepts_valid_dimensions(#[case] raw: &str, #[case] expected: PanelDimensions) {
        let parsed = raw
            .parse::<PanelDimensions>()
            .expect("valid dimensions should parse");
        assert_eq!(expected, parsed);
    }

    #[test]
    fn parse_rejects_missing_separator() {
        let result = "64".parse::<PanelDimensions>();
        assert_matches!(result, Err(PanelDimensionsParseError::InvalidFormat));
    }

    #[test]
    fn parse_rejects_zero_dimensions() {
        let result = "0x64".parse::<PanelDimensions>();
        assert_matches!(result, Err(PanelDimensionsParseError::ZeroDimension));
    }
}
