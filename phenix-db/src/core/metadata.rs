//! # JSONB Metadata Handling and Indexing
//!
//! This module provides utilities for handling structured metadata as JSONB
//! within unified entities, including validation, indexing, and querying support.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
// use std::collections::HashMap;

use crate::core::error::{PhenixDBError, Result};

/// JSONB metadata type alias
pub type JSONB = serde_json::Value;

/// Metadata field path for nested access
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPath(pub Vec<String>);

impl FieldPath {
    /// Create new field path from dot-separated string
    pub fn from_str(path: &str) -> Self {
        Self(path.split('.').map(|s| s.to_string()).collect())
    }

    /// Create field path from vector of strings
    pub fn from_vec(path: Vec<String>) -> Self {
        Self(path)
    }

    /// Get path as dot-separated string
    pub fn to_string(&self) -> String {
        self.0.join(".")
    }

    /// Get path depth
    pub fn depth(&self) -> usize {
        self.0.len()
    }
}

impl From<&str> for FieldPath {
    fn from(path: &str) -> Self {
        Self::from_str(path)
    }
}

impl From<Vec<String>> for FieldPath {
    fn from(path: Vec<String>) -> Self {
        Self::from_vec(path)
    }
}

/// Metadata query operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataQuery {
    /// Exact field value match
    Equals { field: String, value: JSONB },
    /// Field value not equal
    NotEquals { field: String, value: JSONB },
    /// Field value in list
    In { field: String, values: Vec<JSONB> },
    /// Field value not in list
    NotIn { field: String, values: Vec<JSONB> },
    /// Field exists
    Exists { field: String },
    /// Field does not exist
    NotExists { field: String },
    /// Numeric comparison
    GreaterThan { field: String, value: f64 },
    GreaterThanOrEqual { field: String, value: f64 },
    LessThan { field: String, value: f64 },
    LessThanOrEqual { field: String, value: f64 },
    /// String operations
    Contains { field: String, substring: String },
    StartsWith { field: String, prefix: String },
    EndsWith { field: String, suffix: String },
    /// Array operations
    ArrayContains { field: String, value: JSONB },
    ArrayLength { field: String, length: usize },
    /// Logical operations
    And { queries: Vec<MetadataQuery> },
    Or { queries: Vec<MetadataQuery> },
    Not { query: Box<MetadataQuery> },
}

/// Metadata utilities for JSONB operations
pub struct MetadataUtils;

impl MetadataUtils {
    /// Get nested field value from JSONB
    pub fn get_field<'a>(metadata: &'a JSONB, path: &FieldPath) -> Option<&'a JSONB> {
        let mut current = metadata;
        
        for segment in &path.0 {
            match current {
                Value::Object(map) => {
                    current = map.get(segment)?;
                }
                _ => return None,
            }
        }
        
        Some(current)
    }

    /// Set nested field value in JSONB
    pub fn set_field(metadata: &mut JSONB, path: &FieldPath, value: JSONB) -> Result<()> {
        if path.0.is_empty() {
            return Err(PhenixDBError::ValidationError {
                message: "Field path cannot be empty".to_string(),
            });
        }

        // Ensure metadata is an object
        if !metadata.is_object() {
            *metadata = Value::Object(Map::new());
        }

        let mut current = metadata;
        
        // Navigate to parent of target field
        for segment in &path.0[..path.0.len() - 1] {
            current = match current {
                Value::Object(map) => {
                    map.entry(segment.clone())
                        .or_insert_with(|| Value::Object(Map::new()))
                }
                _ => {
                    return Err(PhenixDBError::ValidationError {
                        message: format!("Cannot set field on non-object at path segment: {}", segment),
                    });
                }
            };
        }

        // Set the final field
        if let Value::Object(map) = current {
            map.insert(path.0.last().unwrap().clone(), value);
            Ok(())
        } else {
            Err(PhenixDBError::ValidationError {
                message: "Cannot set field on non-object".to_string(),
            })
        }
    }

    /// Remove field from JSONB
    pub fn remove_field(metadata: &mut JSONB, path: &FieldPath) -> Result<Option<JSONB>> {
        if path.0.is_empty() {
            return Err(PhenixDBError::ValidationError {
                message: "Field path cannot be empty".to_string(),
            });
        }

        let mut current = metadata;
        
        // Navigate to parent of target field
        for segment in &path.0[..path.0.len() - 1] {
            current = match current {
                Value::Object(map) => {
                    map.get_mut(segment).ok_or_else(|| PhenixDBError::ValidationError {
                        message: format!("Field path not found: {}", segment),
                    })?
                }
                _ => {
                    return Err(PhenixDBError::ValidationError {
                        message: format!("Cannot navigate non-object at path segment: {}", segment),
                    });
                }
            };
        }

        // Remove the final field
        if let Value::Object(map) = current {
            Ok(map.remove(path.0.last().unwrap()))
        } else {
            Err(PhenixDBError::ValidationError {
                message: "Cannot remove field from non-object".to_string(),
            })
        }
    }

    /// Check if field exists in JSONB
    pub fn has_field(metadata: &JSONB, path: &FieldPath) -> bool {
        Self::get_field(metadata, path).is_some()
    }

    /// Get all field paths in JSONB (flattened)
    pub fn get_all_paths(metadata: &JSONB) -> Vec<FieldPath> {
        let mut paths = Vec::new();
        Self::collect_paths(metadata, &mut Vec::new(), &mut paths);
        paths
    }

    /// Recursively collect all field paths
    fn collect_paths(value: &JSONB, current_path: &mut Vec<String>, paths: &mut Vec<FieldPath>) {
        match value {
            Value::Object(map) => {
                for (key, val) in map {
                    current_path.push(key.clone());
                    paths.push(FieldPath::from_vec(current_path.clone()));
                    Self::collect_paths(val, current_path, paths);
                    current_path.pop();
                }
            }
            Value::Array(arr) => {
                for (index, val) in arr.iter().enumerate() {
                    current_path.push(index.to_string());
                    paths.push(FieldPath::from_vec(current_path.clone()));
                    Self::collect_paths(val, current_path, paths);
                    current_path.pop();
                }
            }
            _ => {
                // Leaf value, path already added
            }
        }
    }

    /// Validate JSONB metadata
    pub fn validate_metadata(metadata: &JSONB) -> Result<()> {
        // Check maximum depth
        const MAX_DEPTH: usize = 10;
        if Self::get_depth(metadata) > MAX_DEPTH {
            return Err(PhenixDBError::ValidationError {
                message: format!("Metadata depth exceeds maximum of {}", MAX_DEPTH),
            });
        }

        // Check maximum size
        let serialized = serde_json::to_string(metadata)?;
        const MAX_SIZE: usize = 1024 * 1024; // 1MB
        if serialized.len() > MAX_SIZE {
            return Err(PhenixDBError::ValidationError {
                message: format!("Metadata size {} exceeds maximum of {} bytes", serialized.len(), MAX_SIZE),
            });
        }

        // Check for reserved field names
        const RESERVED_FIELDS: &[&str] = &["_id", "_version", "_created_at", "_updated_at"];
        Self::check_reserved_fields(metadata, RESERVED_FIELDS)?;

        Ok(())
    }

    /// Get maximum depth of JSONB structure
    fn get_depth(value: &JSONB) -> usize {
        match value {
            Value::Object(map) => {
                map.values().map(Self::get_depth).max().unwrap_or(0) + 1
            }
            Value::Array(arr) => {
                arr.iter().map(Self::get_depth).max().unwrap_or(0) + 1
            }
            _ => 1,
        }
    }

    /// Check for reserved field names
    fn check_reserved_fields(value: &JSONB, reserved: &[&str]) -> Result<()> {
        match value {
            Value::Object(map) => {
                for key in map.keys() {
                    if reserved.contains(&key.as_str()) {
                        return Err(PhenixDBError::ValidationError {
                            message: format!("Field name '{}' is reserved", key),
                        });
                    }
                }
                
                // Recursively check nested objects
                for val in map.values() {
                    Self::check_reserved_fields(val, reserved)?;
                }
            }
            Value::Array(arr) => {
                for val in arr {
                    Self::check_reserved_fields(val, reserved)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Evaluate metadata query against JSONB
    pub fn evaluate_query(metadata: &JSONB, query: &MetadataQuery) -> bool {
        match query {
            MetadataQuery::Equals { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    field_value == value
                } else {
                    false
                }
            }
            MetadataQuery::NotEquals { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    field_value != value
                } else {
                    true // Field doesn't exist, so it's not equal
                }
            }
            MetadataQuery::In { field, values } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    values.contains(field_value)
                } else {
                    false
                }
            }
            MetadataQuery::NotIn { field, values } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    !values.contains(field_value)
                } else {
                    true // Field doesn't exist, so it's not in the list
                }
            }
            MetadataQuery::Exists { field } => {
                Self::has_field(metadata, &FieldPath::from_str(field))
            }
            MetadataQuery::NotExists { field } => {
                !Self::has_field(metadata, &FieldPath::from_str(field))
            }
            MetadataQuery::GreaterThan { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(num) = field_value.as_f64() {
                        num > *value
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::GreaterThanOrEqual { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(num) = field_value.as_f64() {
                        num >= *value
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::LessThan { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(num) = field_value.as_f64() {
                        num < *value
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::LessThanOrEqual { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(num) = field_value.as_f64() {
                        num <= *value
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::Contains { field, substring } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(s) = field_value.as_str() {
                        s.contains(substring)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::StartsWith { field, prefix } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(s) = field_value.as_str() {
                        s.starts_with(prefix)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::EndsWith { field, suffix } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(s) = field_value.as_str() {
                        s.ends_with(suffix)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::ArrayContains { field, value } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(arr) = field_value.as_array() {
                        arr.contains(value)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::ArrayLength { field, length } => {
                if let Some(field_value) = Self::get_field(metadata, &FieldPath::from_str(field)) {
                    if let Some(arr) = field_value.as_array() {
                        arr.len() == *length
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            MetadataQuery::And { queries } => {
                queries.iter().all(|q| Self::evaluate_query(metadata, q))
            }
            MetadataQuery::Or { queries } => {
                queries.iter().any(|q| Self::evaluate_query(metadata, q))
            }
            MetadataQuery::Not { query } => {
                !Self::evaluate_query(metadata, query)
            }
        }
    }

    /// Merge two JSONB objects
    pub fn merge_metadata(base: &mut JSONB, overlay: &JSONB) -> Result<()> {
        match (base.as_object_mut(), overlay.as_object()) {
            (Some(base_map), Some(overlay_map)) => {
                for (key, value) in overlay_map {
                    match base_map.get_mut(key) {
                        Some(base_value) if base_value.is_object() && value.is_object() => {
                            Self::merge_metadata(base_value, value)?;
                        }
                        _ => {
                            base_map.insert(key.clone(), value.clone());
                        }
                    }
                }
                Ok(())
            }
            _ => {
                *base = overlay.clone();
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_field_path() {
        let path = FieldPath::from_str("user.profile.name");
        assert_eq!(path.0, vec!["user", "profile", "name"]);
        assert_eq!(path.to_string(), "user.profile.name");
        assert_eq!(path.depth(), 3);
    }

    #[test]
    fn test_get_field() {
        let metadata = json!({
            "user": {
                "name": "John",
                "age": 30
            },
            "tags": ["important", "urgent"]
        });

        let name_path = FieldPath::from_str("user.name");
        let name_value = MetadataUtils::get_field(&metadata, &name_path);
        assert_eq!(name_value, Some(&json!("John")));

        let missing_path = FieldPath::from_str("user.email");
        let missing_value = MetadataUtils::get_field(&metadata, &missing_path);
        assert_eq!(missing_value, None);
    }

    #[test]
    fn test_set_field() {
        let mut metadata = json!({});
        let path = FieldPath::from_str("user.profile.name");
        
        MetadataUtils::set_field(&mut metadata, &path, json!("Alice")).unwrap();
        
        let expected = json!({
            "user": {
                "profile": {
                    "name": "Alice"
                }
            }
        });
        assert_eq!(metadata, expected);
    }

    #[test]
    fn test_metadata_queries() {
        let metadata = json!({
            "title": "Test Document",
            "score": 85.5,
            "tags": ["important", "test"],
            "author": {
                "name": "John Doe",
                "age": 30
            }
        });

        // Test equals query
        let equals_query = MetadataQuery::Equals {
            field: "title".to_string(),
            value: json!("Test Document"),
        };
        assert!(MetadataUtils::evaluate_query(&metadata, &equals_query));

        // Test greater than query
        let gt_query = MetadataQuery::GreaterThan {
            field: "score".to_string(),
            value: 80.0,
        };
        assert!(MetadataUtils::evaluate_query(&metadata, &gt_query));

        // Test array contains query
        let array_query = MetadataQuery::ArrayContains {
            field: "tags".to_string(),
            value: json!("important"),
        };
        assert!(MetadataUtils::evaluate_query(&metadata, &array_query));

        // Test nested field query
        let nested_query = MetadataQuery::Equals {
            field: "author.name".to_string(),
            value: json!("John Doe"),
        };
        assert!(MetadataUtils::evaluate_query(&metadata, &nested_query));

        // Test AND query
        let and_query = MetadataQuery::And {
            queries: vec![equals_query, gt_query],
        };
        assert!(MetadataUtils::evaluate_query(&metadata, &and_query));
    }

    #[test]
    fn test_metadata_validation() {
        let valid_metadata = json!({
            "title": "Valid Document",
            "score": 95
        });
        assert!(MetadataUtils::validate_metadata(&valid_metadata).is_ok());

        let invalid_metadata = json!({
            "_id": "reserved_field"
        });
        assert!(MetadataUtils::validate_metadata(&invalid_metadata).is_err());
    }

    #[test]
    fn test_merge_metadata() {
        let mut base = json!({
            "title": "Original",
            "author": {
                "name": "John"
            }
        });

        let overlay = json!({
            "score": 95,
            "author": {
                "age": 30
            }
        });

        MetadataUtils::merge_metadata(&mut base, &overlay).unwrap();

        let expected = json!({
            "title": "Original",
            "score": 95,
            "author": {
                "name": "John",
                "age": 30
            }
        });
        assert_eq!(base, expected);
    }
}