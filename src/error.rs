//! Error handling module for feml library.
//!
//! This module provides a comprehensive error handling system with support for:
//! - DataType-related errors (type mismatches, unsupported operations)
//! - Shape-related errors (dimension mismatches)
//! - Infrastructure errors (I/O, parsing)
//! - Generic error messages with context and path information
//!
//! @author feml contributors
//! @version 0.1.0

use crate::data_type::DataType;
use crate::shape::Shape;
use std::borrow::Cow;
use std::fmt;

/// The underlying kind of error that can occur in the feml library.
///
/// This enum represents all possible error categories that can be raised
/// during tensor operations, data type conversions, and I/O operations.
#[derive(Debug)]
pub enum ErrorKind {
    // ===== DataType =====

    /// Error raised when an unexpected data type is encountered.
    ///
    /// @brief Unexpected data type error.
    /// @param msg Error message describing the context.
    /// @param expected The data type that was expected.
    /// @param got The actual data type that was received.
    UnexpectedDType { msg: &'static str, expected: DataType, got: DataType },

    /// Error raised when a data type is not supported for a specific operation.
    ///
    /// @brief Unsupported data type for operation.
    /// @param dtype The data type that is not supported.
    /// @param op The operation name for which the dtype is unsupported.
    UnsupportedDataTypeForOp { dtype: DataType, op: &'static str },

    // ===== Shape =====

    /// Error raised when a tensor has an unexpected number of dimensions.
    ///
    /// @brief Unexpected number of dimensions error.
    /// @param expected The expected number of dimensions (rank).
    /// @param got The actual number of dimensions received.
    /// @param shape The shape of the tensor that caused the error.
    UnexpectedNumberOfDims { expected: usize, got: usize, shape: Shape },

    // ===== Infra =====

    /// I/O error wrapper.
    ///
    /// @brief I/O operation error.
    /// @param e The underlying std::io::Error.
    Io(std::io::Error),

    /// Integer parsing error wrapper.
    ///
    /// @brief Integer parsing error.
    /// @param e The underlying std::num::ParseIntError.
    ParseInt(std::num::ParseIntError),

    // ===== Runtime =====

    /// Generic error message.
    ///
    /// @brief Generic runtime error message.
    /// @param msg The error message (can be static or owned string).
    Msg(Cow<'static, str>),
}

/// Comprehensive error type with context and backtrace support.
///
/// This struct provides a rich error representation that includes:
/// - The underlying error kind
/// - Additional context information for debugging
/// - Optional file path information
/// - Optional backtrace for debugging
///
/// @note The builder pattern methods (context, with_path) return Self,
///       allowing for fluent chaining of error construction.
///
/// @example
/// ```rust
/// let err = Error::msg("file operation failed")
///     .context("while loading tensor")
///     .with_path("/data/tensor.bin");
/// ```
#[derive(Debug)]
pub struct Error {
    /// The underlying kind of error.
    kind: ErrorKind,
    /// Additional context information providing details about where/why the error occurred.
    context: Vec<Cow<'static, str>>,
    /// Optional file path associated with the error.
    path: Option<std::path::PathBuf>,
    /// Optional backtrace captured at the time of error creation (feature-dependent).
    backtrace: Option<std::backtrace::Backtrace>,
}

impl Error {
    /// Creates a new Error from an ErrorKind.
    ///
    /// @brief Create a new error with the specified kind.
    /// @param kind The underlying error kind.
    /// @return A new Error instance with empty context, no path, and optional backtrace.
    pub fn new(kind: ErrorKind) -> Self {
        Self { kind, context: Vec::new(), path: None, backtrace: capture_backtrace() }
    }

    /// Creates a new Error from a message.
    ///
    /// @brief Create a new error from a message string.
    /// @param msg The error message (can be &str or String).
    /// @return A new Error instance with ErrorKind::Msg.
    ///
    /// @example
    /// ```rust
    /// let err1 = Error::msg("something went wrong");
    /// let err2 = Error::msg(String::from("owned message"));
    /// ```
    pub fn msg(msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Msg(msg.into()))
    }

    /// Adds context information to the error.
    ///
    /// @brief Add context information to the error.
    /// @param ctx Context message describing where the error occurred.
    /// @return Self with the added context, allowing for method chaining.
    ///
    /// @note This method can be called multiple times to build a context chain.
    /// @note Context is displayed in the order it was added (oldest first).
    ///
    /// @example
    /// ```rust
    /// let err = Error::msg("base error")
    ///     .context("in tensor creation")
    ///     .context("during model initialization");
    /// ```
    pub fn context(mut self, ctx: impl Into<Cow<'static, str>>) -> Self {
        self.context.push(ctx.into());
        self
    }

    /// Associates a file path with the error.
    ///
    /// @brief Associate a file path with the error.
    /// @param p The path to associate with the error.
    /// @return Self with the path set, allowing for method chaining.
    ///
    /// @example
    /// ```rust
    /// let err = Error::msg("file not found").with_path("/data/weights.bin");
    /// ```
    pub fn with_path(mut self, p: impl Into<std::path::PathBuf>) -> Self {
        self.path = Some(p.into());
        self
    }
}

/// Captures a backtrace if the backtrace feature is enabled.
///
/// @brief Capture backtrace for debugging.
/// @return Some(Backtrace) if backtrace is enabled and successfully captured, None otherwise.
///
/// @note This function is conditionally compiled based on the "backtrace" feature.
/// @note Even when the feature is enabled, backtrace capture may fail if
///       the backtrace status is not Captured.
fn capture_backtrace() -> Option<std::backtrace::Backtrace> {
    #[cfg(feature = "backtrace")]
    {
        let bt = std::backtrace::Backtrace::capture();
        if matches!(bt.status(), std::backtrace::BacktraceStatus::Captured) {
            return Some(bt);
        }
    }
    None
}

/// Display implementation for Error.
///
/// Formats the error with the following components in order:
/// 1. The root error kind
/// 2. All context messages (one per line, prefixed with "context:")
/// 3. The associated path (if any, prefixed with "path:")
/// 4. The backtrace (if captured)
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 1️ print root error
        write!(f, "{}", self.kind)?;

        // 2 print context
        for ctx in &self.context {
            write!(f, "\ncontext: {ctx}")?;
        }

        // 3 print path
        if let Some(p) = &self.path {
            write!(f, "\npath: {:?}", p)?;
        }

        // 4️ print backtrace
        if let Some(bt) = &self.backtrace {
            write!(f, "\n{bt}")?;
        }

        Ok(())
    }
}

/// Display implementation for ErrorKind.
///
/// Provides human-readable error messages for each error variant.
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedDType { msg, expected, got } => {
                write!(f, "{msg}, expected: {expected:?}, got: {got:?}")
            }

            ErrorKind::UnsupportedDataTypeForOp { dtype, op } => {
                write!(f, "unsupported dtype {dtype:?} for op {op}")
            }

            ErrorKind::UnexpectedNumberOfDims { expected, got, shape } => {
                write!(f, "unexpected rank, expected: {expected}, got: {got} ({shape:?})")
            }

            ErrorKind::Io(e) => write!(f, "{e}"),

            ErrorKind::ParseInt(e) => write!(f, "{e}"),

            ErrorKind::Msg(msg) => write!(f, "{msg}"),
        }
    }
}

/// Implementation of the standard Error trait.
///
/// This enables Error to be used with Rust's error handling infrastructure.
impl std::error::Error for Error {
    /// Returns the underlying source error if one exists.
    ///
    /// @brief Get the underlying source error.
    /// @return Some(source) for Io and ParseInt errors, None otherwise.
    ///
    /// @note Only Io and ParseInt error variants have a source error.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io(e) => Some(e),
            ErrorKind::ParseInt(e) => Some(e),
            _ => None,
        }
    }
}

/// Automatic conversion from std::io::Error to Error.
///
/// @brief Convert from std::io::Error.
/// @param e The I/O error to convert.
/// @return An Error instance wrapping the I/O error.
///
/// @note This enables the `?` operator to work with std::io::Error automatically.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::new(ErrorKind::Io(e))
    }
}

/// Automatic conversion from std::num::ParseIntError to Error.
///
/// @brief Convert from std::num::ParseIntError.
/// @param e The parse error to convert.
/// @return An Error instance wrapping the parse error.
///
/// @note This enables the `?` operator to work with std::num::ParseIntError automatically.
impl From<std::num::ParseIntError> for Error {
    fn from(e: std::num::ParseIntError) -> Self {
        Error::new(ErrorKind::ParseInt(e))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;
    use std::io;

    // Test Error::new()
    #[test]
    fn test_error_new() {
        let err = Error::new(ErrorKind::Msg("test error".into()));
        assert!(matches!(err.kind, ErrorKind::Msg(_)));
        assert!(err.context.is_empty());
        assert!(err.path.is_none());
    }

    // Test Error::msg()
    #[test]
    fn test_error_msg() {
        let err = Error::msg("static string");
        assert!(matches!(err.kind, ErrorKind::Msg(_)));

        let err = Error::msg(String::from("owned string"));
        assert!(matches!(err.kind, ErrorKind::Msg(_)));
    }

    // Test Error::context() - chain multiple contexts
    #[test]
    fn test_error_context() {
        let err = Error::msg("base error")
            .context("first context")
            .context("second context");

        assert_eq!(err.context.len(), 2);
        assert_eq!(err.context[0], "first context");
        assert_eq!(err.context[1], "second context");
    }

    // Test Error::with_path()
    #[test]
    fn test_error_with_path() {
        let err = Error::msg("file error").with_path("/tmp/test.txt");
        assert_eq!(err.path, Some(std::path::PathBuf::from("/tmp/test.txt")));
    }

    // Test chained builder pattern
    #[test]
    fn test_error_builder_chain() {
        let err = Error::msg("operation failed")
            .context("while processing tensor")
            .context("in forward pass")
            .with_path("/model/weights.bin");

        assert_eq!(err.context.len(), 2);
        assert_eq!(err.path, Some(std::path::PathBuf::from("/model/weights.bin")));
    }

    // Test Display for UnexpectedDType
    #[test]
    fn test_display_unexpected_dtype() {
        let kind = ErrorKind::UnexpectedDType {
            msg: "type mismatch",
            expected: DataType::F32,
            got: DataType::I32,
        };
        let s = format!("{kind}");
        assert!(s.contains("type mismatch"));
        assert!(s.contains("F32") || s.contains("I32"));
    }

    // Test Display for UnsupportedDataTypeForOp
    #[test]
    fn test_display_unsupported_dtype() {
        let kind = ErrorKind::UnsupportedDataTypeForOp {
            dtype: DataType::F16,
            op: "conv2d",
        };
        let s = format!("{kind}");
        assert!(s.contains("F16") || s.contains("conv2d"));
        assert!(s.contains("unsupported"));
    }

    // Test Display for UnexpectedNumberOfDims
    #[test]
    fn test_display_unexpected_dims() {
        let shape = Shape([1, 3, 224, 224]);
        let kind = ErrorKind::UnexpectedNumberOfDims {
            expected: 3,
            got: 4,
            shape,
        };
        let s = format!("{kind}");
        assert!(s.contains("unexpected rank"));
        assert!(s.contains("expected: 3"));
        assert!(s.contains("got: 4"));
    }

    // Test Display for Io error
    #[test]
    fn test_display_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let kind = ErrorKind::Io(io_err);
        let s = format!("{kind}");
        assert!(s.contains("file not found"));
    }

    // Test Display for ParseInt error
    #[test]
    fn test_display_parse_int() {
        let parse_err = "abc".parse::<i32>().unwrap_err();
        let kind = ErrorKind::ParseInt(parse_err);
        let s = format!("{kind}");
        assert!(!s.is_empty());
    }

    // Test Display for Msg
    #[test]
    fn test_display_msg() {
        let kind = ErrorKind::Msg("custom error message".into());
        let s = format!("{kind}");
        assert_eq!(s, "custom error message");
    }

    // Test full Error Display with context and path
    #[test]
    fn test_display_full_error() {
        let err = Error::msg("base error")
            .context("context 1")
            .context("context 2")
            .with_path("/test/path");
        let s = format!("{err}");
        assert!(s.contains("base error"));
        assert!(s.contains("context: context 1"));
        assert!(s.contains("context: context 2"));
        assert!(s.contains("path: \"/test/path\""));
    }

    // Test Error source() for Io
    #[test]
    fn test_error_source_io() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let err = Error::new(ErrorKind::Io(io_err));
        assert!(err.source().is_some());
    }

    // Test Error source() for ParseInt
    #[test]
    fn test_error_source_parse_int() {
        let parse_err = "xyz".parse::<i32>().unwrap_err();
        let err = Error::new(ErrorKind::ParseInt(parse_err));
        assert!(err.source().is_some());
    }

    // Test Error source() returns None for Msg
    #[test]
    fn test_error_source_none() {
        let err = Error::msg("no source error");
        assert!(err.source().is_none());
    }

    // Test From<io::Error>
    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::Other, "io error");
        let err: Error = io_err.into();
        assert!(matches!(err.kind, ErrorKind::Io(_)));
    }

    // Test From<ParseIntError>
    #[test]
    fn test_from_parse_int_error() {
        let parse_err = "not a number".parse::<i32>().unwrap_err();
        let err: Error = parse_err.into();
        assert!(matches!(err.kind, ErrorKind::ParseInt(_)));
    }

    // Test backtrace is captured when feature is enabled
    #[test]
    fn test_backtrace_captured() {
        let err = Error::msg("test");
        // When backtrace feature is enabled, backtrace may be captured
        // This test just verifies the field exists and is Option
        let _ = err.backtrace;
    }
}
