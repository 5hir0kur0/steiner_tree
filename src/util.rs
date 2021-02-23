use std::error::Error;

/// Result with boxed error as trait object.
pub type GenericResult<T> = Result<T, Box<dyn Error + Send + Sync>>;
