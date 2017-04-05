use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;
use toml::Value;

pub fn toml_string_from_file(filename: &str) -> String {
    let path = PathBuf::from(filename);
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(..) => panic!("Cannot open TOML file {}", path.display()),
    };
    let mut data = String::new();
    file.read_to_string(&mut data).expect("Can't read TOML file!");
    data
}

pub fn toml_value_from_string(data: &str) -> Value {
    data.parse::<Value>().expect("Invalid TOML!")
}

pub fn toml_value_from_file(filename: &str) -> Value {
    let toml_str = toml_string_from_file(filename);
    toml_value_from_string(&toml_str)
}

pub fn get_value_in_table<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    match *value {
        Value::Table(ref table) => {
            // 'toml' just panics upon reading an invalid key index, so this is needed.
            if !table.contains_key(key) {
                None
            } else {
                Some(&table[key])
            }
        }
        _ => None
    }
}

/// Gets the value of the key in the given TOML table.
pub fn get_toml_value<T: Deserialize>(value: &Value, table_name: &str, key: &str) -> Option<T> {
    match get_value_in_table(value, table_name) {
        Some(table) => match get_value_in_table(&table, key) {
            Some(val) => val.clone().try_into::<T>().ok(),
            None => None,
        },
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get() {
        let val = toml_value_from_string("
[table]
thing=1
dood=true");
        let res: Option<i32> = get_toml_value(&val, "table", "thing");
        assert!(res.is_some());
        let res: Option<bool> = get_toml_value(&val, "table", "none");
        assert!(res.is_none());
        let res: Option<i32> = get_toml_value(&val, "whee", "thing");
        assert!(res.is_none());
        let res: Option<bool> = get_toml_value(&val, "whee", "none");
        assert!(res.is_none());
    }
}
