//! Metadata handling (JSONB)

use serde::{Deserialize, Serialize};

/// Metadata type alias (JSONB)
pub type Metadata = serde_json::Value;

/// Metadata filter for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Filter expression
    pub expression: FilterExpression,
}

/// Filter expression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    /// Equality comparison
    Eq { field: String, value: serde_json::Value },
    /// Not equal comparison
    Ne { field: String, value: serde_json::Value },
    /// Greater than
    Gt { field: String, value: serde_json::Value },
    /// Less than
    Lt { field: String, value: serde_json::Value },
    /// In array
    In { field: String, values: Vec<serde_json::Value> },
    /// Contains
    Contains { field: String, value: serde_json::Value },
    /// Logical AND
    And { expressions: Vec<FilterExpression> },
    /// Logical OR
    Or { expressions: Vec<FilterExpression> },
    /// Logical NOT
    Not { expression: Box<FilterExpression> },
}
