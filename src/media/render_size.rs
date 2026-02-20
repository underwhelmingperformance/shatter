use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use thiserror::Error;

use crate::media::{PanelDimensions, PanelDimensionsParseError};

/// Output sizing strategy for transition renders.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum RenderSize {
    /// Selects the smaller input image dimensions by pixel area.
    #[default]
    Auto,
    /// Forces output to the provided dimensions.
    Fixed(PanelDimensions),
}

impl RenderSize {
    /// Resolves concrete output dimensions for two input images.
    ///
    /// ```
    /// use shatter::{PanelDimensions, RenderSize};
    ///
    /// let from = PanelDimensions::new(128, 64).expect("128x64 should be valid");
    /// let to = PanelDimensions::new(64, 64).expect("64x64 should be valid");
    /// assert_eq!(to, RenderSize::Auto.resolve(from, to));
    /// ```
    #[must_use]
    pub const fn resolve(self, from: PanelDimensions, to: PanelDimensions) -> PanelDimensions {
        match self {
            Self::Auto => {
                if from.area() <= to.area() {
                    from
                } else {
                    to
                }
            }
            Self::Fixed(dimensions) => dimensions,
        }
    }
}

impl Display for RenderSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Fixed(dimensions) => dimensions.fmt(f),
        }
    }
}

impl FromStr for RenderSize {
    type Err = RenderSizeParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }

        value
            .parse::<PanelDimensions>()
            .map(Self::Fixed)
            .map_err(|source| {
                if matches!(source, PanelDimensionsParseError::InvalidFormat) {
                    RenderSizeParseError::InvalidValue {
                        value: value.to_string(),
                    }
                } else {
                    RenderSizeParseError::InvalidDimensions { source }
                }
            })
    }
}

/// Errors returned while parsing a render size strategy.
#[derive(Debug, Error)]
pub enum RenderSizeParseError {
    /// The size is neither `auto` nor `WIDTHxHEIGHT`.
    #[error("size must be `auto` or WIDTHxHEIGHT, got `{value}`")]
    InvalidValue {
        /// Original input.
        value: String,
    },
    /// A `WIDTHxHEIGHT` value was supplied but failed to parse.
    #[error(transparent)]
    InvalidDimensions {
        /// Underlying dimensions parse error.
        source: PanelDimensionsParseError,
    },
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn parse_accepts_auto_and_fixed_sizes() -> anyhow::Result<()> {
        let fixed = "32x16".parse::<RenderSize>()?;
        assert_eq!(
            RenderSize::Fixed(PanelDimensions::new(32, 16).expect("32x16 should be valid")),
            fixed
        );

        let auto = "AUTO".parse::<RenderSize>()?;
        assert_eq!(RenderSize::Auto, auto);

        Ok(())
    }

    #[test]
    fn parse_rejects_unknown_size_keyword() {
        let result = "small".parse::<RenderSize>();
        assert_matches!(
            result,
            Err(RenderSizeParseError::InvalidValue { ref value }) if value == "small"
        );
    }
}
