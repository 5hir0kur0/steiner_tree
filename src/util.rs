use std::cmp::Ordering;
use std::error::Error;
use std::ops::Add;

/// Result with boxed error as trait object.
pub type GenericResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Numerical type that can either be an unsigned number or positive infinity.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NaturalOrInfinite(i64);

impl PartialOrd for NaturalOrInfinite {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0 < 0 && other.0 < 0 {
            Some(Ordering::Equal)
        } else if other.0 < 0 {
            Some(Ordering::Less)
        } else if self.0 < 0 {
            Some(Ordering::Greater)
        } else {
            Some(self.0.cmp(&other.0))
        }
    }
}

impl Add for NaturalOrInfinite {
    type Output = NaturalOrInfinite;

    fn add(self, rhs: Self) -> Self::Output {
        if self.0 < 0 || rhs.0 < 0 {
            NaturalOrInfinite(-1)
        } else {
            NaturalOrInfinite(self.0 + rhs.0)
        }
    }
}

impl NaturalOrInfinite {
    pub fn infinity() -> Self {
        NaturalOrInfinite(-1)
    }
}

impl From<u32> for NaturalOrInfinite {
    fn from(val: u32) -> Self {
        NaturalOrInfinite(val as i64)
    }
}

pub type TestResult = GenericResult<()>;
