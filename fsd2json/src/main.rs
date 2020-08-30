use byteorder::{LittleEndian, ReadBytesExt};
use clap::{App, Arg};
use log::{debug, info, trace};
use std::fmt;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

fn pickle_to_json(v: &serde_pickle::Value) -> serde_json::Value {
    match v {
        serde_pickle::Value::Bool(v) => (*v).into(),
        serde_pickle::Value::I64(v) => (*v).into(),
        serde_pickle::Value::Int(_) => panic!("BigInt not supported"),
        serde_pickle::Value::F64(v) => (*v).into(),
        serde_pickle::Value::Bytes(v) => std::str::from_utf8(v).unwrap().into(),
        serde_pickle::Value::String(v) => (*v).clone().into(),
        serde_pickle::Value::List(v) => (*v)
            .iter()
            .map(|x| pickle_to_json(x))
            .collect::<Vec<serde_json::Value>>()
            .into(),
        serde_pickle::Value::Tuple(v) => (*v)
            .iter()
            .map(|x| pickle_to_json(x))
            .collect::<Vec<serde_json::Value>>()
            .into(),
        serde_pickle::Value::Dict(v) => {
            //
            //
            v.iter()
                .map(|(k, v)| {
                    let k = pickle_to_json(&k.clone().into_value());
                    let v = pickle_to_json(v);
                    assert!(k.is_string());
                    (k.as_str().unwrap().to_string(), v)
                })
                .collect::<serde_json::map::Map<String, serde_json::Value>>()
                .into()
        }
        _ => panic!("Unsupported value"),
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum FooterKeyType {
    OffsetIntWithSize,
    OffsetLongWithSize,
    OffsetInt,
    OffsetLong,
    Arbitrary,
}

#[derive(Debug)]
struct DictFooter {
    schema: serde_json::Value,
    buffer: Vec<u8>,
    key_type: FooterKeyType,
    size: u32,
    item_size: u64,
    start_offset: u64,
}

#[derive(Debug)]
struct FooterEntry {
    pub key: FsdValue,
    pub offset: u32,
    pub size: Option<u32>,
}

impl<'a> IntoIterator for &'a DictFooter {
    type Item = FooterEntry;
    type IntoIter = FooterIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        FooterIterator {
            footer: self,
            index: 0,
        }
    }
}

struct FooterIterator<'a> {
    footer: &'a DictFooter,
    index: usize,
}

impl<'a> Iterator for FooterIterator<'a> {
    type Item = FooterEntry;
    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.index > self.footer.size as usize {
            None
        } else {
            let mut reader = std::io::Cursor::new(&self.footer.buffer);

            match self.footer.key_type {
                FooterKeyType::Arbitrary => {
                    reader
                        .seek(std::io::SeekFrom::Start(
                            (self.index - 1) as u64 * self.footer.item_size
                                + self.footer.start_offset,
                        ))
                        .unwrap();
                    let n = reader.read_u32::<LittleEndian>().unwrap();
                    let data_offset_from_object_start = self.footer.start_offset + n as u64;
                    //
                    let key = FsdValue::from_buffer(
                        &mut reader,
                        data_offset_from_object_start - 4,
                        &self.footer.schema["keyFooter"]["itemTypes"],
                    )
                    .unwrap();

                    Some(FooterEntry {
                        key: match &key {
                            FsdValue::Object(o) => o["key"].clone(),
                            _ => key.clone(),
                        },
                        size: match &key {
                            FsdValue::Object(o) => match o["size"] {
                                FsdValue::UInt32(v) => Some(v),
                                _ => None,
                            },
                            _ => None,
                        },
                        offset: match &key {
                            FsdValue::Object(o) => match o["offset"] {
                                FsdValue::UInt32(v) => v,
                                _ => 0,
                            },
                            _ => 0,
                        },
                    })
                }
                _ => {
                    reader
                        .seek(std::io::SeekFrom::Start(
                            (self.index - 1) as u64 * self.footer.item_size
                                + self.footer.start_offset,
                        ))
                        .unwrap();

                    let key = if self.footer.key_type == FooterKeyType::OffsetInt
                        || self.footer.key_type == FooterKeyType::OffsetIntWithSize
                    {
                        reader.read_u32::<LittleEndian>().unwrap() as u64
                    } else {
                        reader.read_u64::<LittleEndian>().unwrap()
                    };

                    let offset = reader.read_u32::<LittleEndian>().unwrap();

                    let size = if self.footer.key_type == FooterKeyType::OffsetIntWithSize
                        || self.footer.key_type == FooterKeyType::OffsetLongWithSize
                    {
                        Some(reader.read_u32::<LittleEndian>().unwrap())
                    } else {
                        None
                    };
                    Some(Self::Item {
                        key: key.into(),
                        offset,
                        size,
                    })
                }
            }
        }
    }
}

impl DictFooter {
    pub fn new(buffer: Vec<u8>, schema: &serde_json::Value) -> Self {
        let mut reader = BufReader::new(&buffer[..]);
        let size = reader.read_u32::<LittleEndian>().unwrap();

        let key_type = schema["keyTypes"]["type"].as_str().unwrap().to_string();
        if key_type == "int" || key_type == "long" {
            // TODO(alexander): Handle other key types
            let is_int_key = schema["keyTypes"]["type"].as_str().unwrap() == "int";
            let has_size = !schema["keyFooter"]["itemTypes"]["attributes"]["size"].is_null();
            let key_type = if is_int_key {
                if has_size {
                    FooterKeyType::OffsetIntWithSize
                } else {
                    FooterKeyType::OffsetInt
                }
            } else {
                if has_size {
                    FooterKeyType::OffsetLongWithSize
                } else {
                    FooterKeyType::OffsetLong
                }
            };

            let item_size = match key_type {
                FooterKeyType::OffsetInt => std::mem::size_of::<u32>() + std::mem::size_of::<u32>(),
                FooterKeyType::OffsetIntWithSize => {
                    std::mem::size_of::<u32>()
                        + std::mem::size_of::<u32>()
                        + std::mem::size_of::<u32>()
                }
                FooterKeyType::OffsetLong => {
                    std::mem::size_of::<u64>() + std::mem::size_of::<u32>()
                }
                FooterKeyType::OffsetLongWithSize => {
                    std::mem::size_of::<u64>()
                        + std::mem::size_of::<u32>()
                        + std::mem::size_of::<u32>()
                }
                _ => 0,
            } as u64;

            Self {
                schema: schema.clone(),
                buffer,
                key_type,
                size,
                item_size,
                start_offset: 4,
            }
        } else {
            Self {
                schema: schema.clone(),
                buffer,
                key_type: FooterKeyType::Arbitrary,
                size,
                item_size: 4,
                start_offset: 4,
            }
        }
    }
}

fn as_u32_le(array: &[u8]) -> u32 {
    ((array[0] as u32) << 0)
        | ((array[1] as u32) << 8)
        | ((array[2] as u32) << 16)
        | ((array[3] as u32) << 24)
}

#[derive(Debug)]
pub enum FsdDecodeError {
    UnsupportedType(String),
    IoError(std::io::Error),
    Utf8Error(std::str::Utf8Error),
}

impl fmt::Display for FsdDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<std::io::Error> for FsdDecodeError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<std::str::Utf8Error> for FsdDecodeError {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Utf8Error(err)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FsdValue {
    None,
    Float(f32),
    Double(f64),
    Bool(bool),
    UInt64(u64),
    Int64(i64),
    UInt32(u32),
    Int32(i32),
    Long(u64),
    Object(std::collections::HashMap<String, FsdValue>),
    List(Vec<FsdValue>),
    String(String),
    Dict(std::collections::HashMap<u64, FsdValue>),
    Map(std::collections::HashMap<FsdValue, FsdValue>),
}

impl Eq for FsdValue {}

impl std::hash::Hash for FsdValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::None => 0.hash(state),
            Self::Float(f) => f.to_le_bytes().hash(state),
            Self::Double(f) => f.to_le_bytes().hash(state),
            Self::Bool(f) => f.hash(state),
            Self::UInt64(f) => f.hash(state),
            Self::Int64(f) => f.hash(state),
            Self::UInt32(f) => f.hash(state),
            Self::Int32(f) => f.hash(state),
            Self::Long(f) => f.hash(state),
            Self::Object(_) => 0.hash(state),
            Self::List(f) => f.hash(state),
            Self::String(f) => f.hash(state),
            Self::Dict(_) => 0.hash(state),
            Self::Map(_) => 0.hash(state),
        };
    }
}

impl From<f32> for FsdValue {
    fn from(f: f32) -> Self {
        Self::Float(f)
    }
}

impl From<f64> for FsdValue {
    fn from(f: f64) -> Self {
        Self::Double(f)
    }
}

impl From<bool> for FsdValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i32> for FsdValue {
    fn from(i: i32) -> Self {
        Self::Int32(i)
    }
}

impl From<u32> for FsdValue {
    fn from(i: u32) -> Self {
        Self::UInt32(i)
    }
}

impl From<u64> for FsdValue {
    fn from(i: u64) -> Self {
        Self::UInt64(i)
    }
}

impl From<i64> for FsdValue {
    fn from(i: i64) -> Self {
        Self::Int64(i)
    }
}

impl From<String> for FsdValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<std::collections::HashMap<u64, FsdValue>> for FsdValue {
    fn from(d: std::collections::HashMap<u64, FsdValue>) -> Self {
        Self::Dict(d)
    }
}

impl From<std::collections::HashMap<String, FsdValue>> for FsdValue {
    fn from(o: std::collections::HashMap<String, FsdValue>) -> Self {
        Self::Object(o)
    }
}

impl From<std::collections::HashMap<FsdValue, FsdValue>> for FsdValue {
    fn from(o: std::collections::HashMap<FsdValue, FsdValue>) -> Self {
        Self::Map(o)
    }
}

impl From<Vec<FsdValue>> for FsdValue {
    fn from(v: Vec<FsdValue>) -> Self {
        Self::List(v)
    }
}

impl From<serde_json::Value> for FsdValue {
    fn from(v: serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => FsdValue::None,
            serde_json::Value::Bool(b) => b.into(),
            serde_json::Value::Array(a) => a
                .into_iter()
                .map(|x| x.into())
                .collect::<Vec<FsdValue>>()
                .into(),
            serde_json::Value::Number(n) => {
                if n.is_f64() {
                    n.as_f64().unwrap().into()
                } else if n.is_i64() {
                    n.as_i64().unwrap().into()
                } else if n.is_u64() {
                    n.as_u64().unwrap().into()
                } else {
                    Self::None
                }
            }
            serde_json::Value::String(s) => s.into(),
            serde_json::Value::Object(m) => m
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect::<std::collections::HashMap<String, FsdValue>>()
                .into(),
        }
    }
}

impl Into<serde_json::Value> for FsdValue {
    fn into(self) -> serde_json::Value {
        match self {
            Self::None => serde_json::Value::Null,
            Self::Float(f) => f.into(),
            Self::Double(f) => f.into(),
            Self::Bool(f) => f.into(),
            Self::UInt64(f) => f.into(),
            Self::Int64(f) => f.into(),
            Self::UInt32(f) => f.into(),
            Self::Int32(f) => f.into(),
            Self::Long(f) => f.into(),
            Self::Object(f) => f
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into(),
            Self::List(f) => f.into(),
            Self::String(f) => f.into(),
            Self::Dict(f) => f
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.into()))
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into(),
            Self::Map(f) => f
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.into()))
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into(),
        }
    }
}

impl std::string::ToString for FsdValue {
    fn to_string(&self) -> String {
        match self {
            Self::None => "null".into(),
            Self::Float(f) => f.to_string(),
            Self::Double(f) => f.to_string(),
            Self::Bool(f) => f.to_string(),
            Self::UInt64(f) => f.to_string(),
            Self::Int64(f) => f.to_string(),
            Self::UInt32(f) => f.to_string(),
            Self::Int32(f) => f.to_string(),
            Self::Long(f) => f.to_string(),
            Self::String(f) => f.to_string(),
            _ => format!("{:?}", self),
        }
    }
}

impl FsdValue {
    pub fn from_buffer(
        mut buffer: &mut std::io::Cursor<&Vec<u8>>,
        offset: u64,
        schema: &serde_json::Value,
    ) -> Result<Self, FsdDecodeError> {
        let s_type = schema["type"].as_str().unwrap().to_string();
        debug!("Parsing {} at {}", s_type, offset);
        let is_double_precision = schema
            .get("precision")
            .map(|x| x.as_str().unwrap_or("single"))
            .unwrap_or("single")
            == "double";
        let is_unsigned = {
            let min = schema.get("min");
            let exclusive_min = schema.get("exclusiveMin");
            min.is_some() && min.map(|x| x.as_i64().unwrap_or(0)).unwrap_or(0) >= 0
                || exclusive_min.is_some()
                    && exclusive_min.map(|x| x.as_i64().unwrap_or(0)).unwrap_or(0) >= -1
        };
        let is_fixed_size = schema.get("fixedItemSize").is_some();
        let aliases = schema.get("aliases");

        // Jump to the start offset of the data
        buffer.seek(std::io::SeekFrom::Start(offset))?;

        // Map some types to a more primitive type
        let s_type = {
            match s_type.as_str() {
                "typeID" => "int",
                "localizationID" => "int",
                "npcTag" => "int",
                "deploymentType" => "int",
                "npcEnemyFleetTypeID" => "int",
                "groupBehaviorTreeID" => "int",
                "npcCorporationID" => "int",
                "spawnTableID" => "int",
                "npcFleetCounterTableID" => "int",
                "dungeonID" => "int",
                "typeListID" => "int",
                "npcFleetTypeID" => "int",
                _ => s_type.as_str(),
            }
        };
        match s_type {
            "string" => {
                let count = buffer.read_u32::<LittleEndian>()?;
                let mut str_buffer = vec![0; count as usize];
                buffer.read_exact(&mut str_buffer)?;

                Ok(std::str::from_utf8(&str_buffer)?.to_string().into())
            }
            "float" => {
                // TODO(alexander): We really really should have the schema pre-parsed into some kind of structure
                // This is annoying and stupid to deal with
                if is_double_precision {
                    Ok(buffer.read_f64::<LittleEndian>()?.into())
                } else {
                    Ok(buffer.read_f32::<LittleEndian>()?.into())
                }
            }
            "bool" => Ok((buffer.read_u8()? == 255).into()),
            "int" => {
                if is_unsigned {
                    Ok(buffer.read_u32::<LittleEndian>()?.into())
                } else {
                    Ok(buffer.read_i32::<LittleEndian>()?.into())
                }
            }
            "object" => {
                // TODO(alexander): What a horrible mess this is...
                let mut output = std::collections::HashMap::<String, FsdValue>::new();
                let has_size = schema.get("size").is_some();
                let mut variable_data_offset_base = 0u64;
                let mut offset_attributes_lookup_table = std::collections::HashMap::new();
                if !has_size {
                    let mut offset_attributes: Vec<String> = schema
                        ["attributesWithVariableOffsets"]
                        .as_array()
                        .unwrap_or(&Vec::new())
                        .iter()
                        .map(|x| x.as_str().unwrap().to_string())
                        .collect();

                    let optional_value_lookups =
                        schema["optionalValueLookups"].as_object().unwrap();
                    let end_of_fixed_size_data = schema["endOfFixedSizeData"].as_u64().unwrap_or(0);
                    if !optional_value_lookups.is_empty() {
                        buffer
                            .seek(std::io::SeekFrom::Start(offset + end_of_fixed_size_data))
                            .unwrap();
                        let optional_attributes_field = buffer.read_u64::<LittleEndian>().unwrap();
                        for (_, (k, v)) in optional_value_lookups.iter().enumerate() {
                            //
                            let i = v.as_u64().unwrap();
                            if optional_attributes_field & i as u64 == 0 {
                                offset_attributes.retain(|x| x != k);
                            }
                        }
                    }
                    let offset_attribute_array_start = offset + end_of_fixed_size_data + 8;
                    let offset_attribute_offsets_type_size = 4 * offset_attributes.len();
                    variable_data_offset_base =
                        offset_attribute_array_start + offset_attribute_offsets_type_size as u64;

                    buffer
                        .seek(std::io::SeekFrom::Start(offset_attribute_array_start))
                        .unwrap();
                    let mut offsets_data = vec![0; offset_attribute_offsets_type_size];
                    buffer.read_exact(&mut offsets_data)?;
                    let offset_table: Vec<u32> = offsets_data
                        .chunks_exact(4)
                        .into_iter()
                        .map(|a| as_u32_le(a))
                        .collect();
                    for (k, v) in offset_attributes.iter().zip(offset_table.iter()) {
                        trace!(
                            "Loading offset lookup for {} -> {}",
                            k,
                            variable_data_offset_base + (*v) as u64
                        );
                        offset_attributes_lookup_table.insert(k.clone(), *v);
                    }
                }

                let attributes = schema["attributes"].as_object().unwrap();
                for (k, attribute_schema) in attributes.iter() {
                    //
                    let v = if schema["constantAttributeOffsets"].get(k).is_some() {
                        trace!(
                            "Parsing object field ({}) with constant offset {}",
                            k,
                            schema["constantAttributeOffsets"][k]
                        );
                        Some(FsdValue::from_buffer(
                            &mut buffer,
                            offset + schema["constantAttributeOffsets"][k].as_u64().unwrap(),
                            attribute_schema,
                        )?)
                    } else {
                        //
                        if !offset_attributes_lookup_table.contains_key(k) {
                            trace!("Check for default value for {}", k);
                            if attribute_schema.get("default").is_some() {
                                Some(attribute_schema["default"].clone().into())
                            } else {
                                None
                            }
                        } else {
                            trace!(
                                "Parsing object field ({}) with offset lookup {}",
                                k,
                                offset_attributes_lookup_table[k]
                            );
                            Some(FsdValue::from_buffer(
                                &mut buffer,
                                variable_data_offset_base
                                    + offset_attributes_lookup_table[k] as u64,
                                attribute_schema,
                            )?)
                        }
                    };
                    match v {
                        Some(v) => {
                            output.insert(k.clone(), v);
                        }
                        None => {
                            // None is no default value but has lookup table entry
                            // So we ignore
                            // Because OPTIONAL field
                        }
                    }
                }

                Ok(output.into())
            }
            "list" => {
                let fixed_length = false;
                let count = buffer.read_u32::<LittleEndian>().unwrap();
                let count_offset = if fixed_length { 0 } else { 4 };
                let schema = &schema["itemTypes"];

                let mut output = Vec::new();
                if is_fixed_size {
                    let item_size = schema["size"].as_u64().unwrap();

                    for i in 0..count {
                        let total_offset = offset + count_offset + item_size * i as u64;
                        output.push(FsdValue::from_buffer(buffer, total_offset, &schema)?);
                    }
                } else {
                    for i in 0..count {
                        //
                        buffer.seek(std::io::SeekFrom::Start(
                            offset + count_offset + 4 * i as u64,
                        ))?;
                        let n = buffer.read_u32::<LittleEndian>().unwrap();
                        let data_offset_from_object_start = offset + n as u64;
                        output.push(FsdValue::from_buffer(
                            buffer,
                            data_offset_from_object_start,
                            &schema,
                        )?);
                    }
                }

                Ok(output.into())
            }
            "dict" => {
                trace!("Read Footer");
                let size_of_data = buffer.read_u32::<LittleEndian>()?;
                let offset_to_data = buffer.seek(std::io::SeekFrom::Current(0))?;

                buffer.seek(std::io::SeekFrom::Start(offset + size_of_data as u64))?; // Jump to the end of the data, which is where the footer size is
                let size_of_footer = buffer.read_u32::<LittleEndian>()?;
                trace!("Jump to footer start");
                buffer.seek(std::io::SeekFrom::Start(
                    offset + size_of_data as u64 - size_of_footer as u64,
                ))?; // // Jump to the start of the footer // offset_to_data + size_of_data as u64 - size_of_footer as u64,

                trace!("Load footer buffer");
                let mut footer_buffer = vec![0; size_of_footer as usize];
                buffer.read_exact(&mut footer_buffer)?;

                trace!(
                    "Parse footer as {} with size {}",
                    offset + size_of_data as u64 - size_of_footer as u64,
                    size_of_footer
                );
                let footer = DictFooter::new(footer_buffer, &schema);

                trace!("Read values");
                let value_schema = &schema["valueTypes"];
                let mut output = std::collections::HashMap::<FsdValue, FsdValue>::new();
                for n in footer.into_iter() {
                    //
                    trace!("Read {:?} at {}", n.key, offset_to_data + n.offset as u64);
                    buffer
                        .seek(std::io::SeekFrom::Start(offset_to_data + n.offset as u64))
                        .unwrap();
                    // TODO(alexander): Figure out if we need size somehow
                    // so far it all seems to work even if we just ignore it :)
                    output.insert(
                        n.key,
                        match n.size {
                            Some(_) => FsdValue::from_buffer(
                                buffer,
                                offset_to_data + n.offset as u64,
                                &value_schema,
                            )?,
                            None => FsdValue::from_buffer(
                                buffer,
                                offset_to_data + n.offset as u64,
                                &value_schema,
                            )?,
                        },
                    );
                }
                Ok(output.into())
            }
            "long" => Ok(buffer.read_u64::<LittleEndian>()?.into()),
            // TODO(alexander): Merge the vector things somehow
            "vector4" => {
                //
                Ok(if is_double_precision {
                    let data: Vec<FsdValue> = [
                        buffer.read_f64::<LittleEndian>()?.into(),
                        buffer.read_f64::<LittleEndian>()?.into(),
                        buffer.read_f64::<LittleEndian>()?.into(),
                        buffer.read_f64::<LittleEndian>()?.into(),
                    ]
                    .to_vec();
                    match aliases {
                        Some(aliases) => {
                            let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                            let aliases = aliases.as_object().unwrap();
                            for (k, v) in aliases.into_iter() {
                                out_data
                                    .insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                            }
                            out_data.into()
                        }
                        None => data.into(),
                    }
                } else {
                    let data: Vec<FsdValue> = [
                        buffer.read_f32::<LittleEndian>()?.into(),
                        buffer.read_f32::<LittleEndian>()?.into(),
                        buffer.read_f32::<LittleEndian>()?.into(),
                        buffer.read_f32::<LittleEndian>()?.into(),
                    ]
                    .to_vec();
                    match aliases {
                        Some(aliases) => {
                            let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                            let aliases = aliases.as_object().unwrap();
                            for (k, v) in aliases.into_iter() {
                                out_data
                                    .insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                            }
                            out_data.into()
                        }
                        None => data.into(),
                    }
                })
            }
            "vector3" => Ok(if is_double_precision {
                let data: Vec<FsdValue> = [
                    buffer.read_f64::<LittleEndian>()?.into(),
                    buffer.read_f64::<LittleEndian>()?.into(),
                    buffer.read_f64::<LittleEndian>()?.into(),
                ]
                .to_vec();
                match aliases {
                    Some(aliases) => {
                        let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                        let aliases = aliases.as_object().unwrap();
                        for (k, v) in aliases.into_iter() {
                            out_data.insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                        }
                        out_data.into()
                    }
                    None => data.into(),
                }
            } else {
                let data: Vec<FsdValue> = [
                    buffer.read_f32::<LittleEndian>()?.into(),
                    buffer.read_f32::<LittleEndian>()?.into(),
                    buffer.read_f32::<LittleEndian>()?.into(),
                ]
                .to_vec();
                match aliases {
                    Some(aliases) => {
                        let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                        let aliases = aliases.as_object().unwrap();
                        for (k, v) in aliases.into_iter() {
                            out_data.insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                        }
                        out_data.into()
                    }
                    None => data.into(),
                }
            }),
            "vector2" => Ok(if is_double_precision {
                let data: Vec<FsdValue> = [
                    buffer.read_f64::<LittleEndian>()?.into(),
                    buffer.read_f64::<LittleEndian>()?.into(),
                ]
                .to_vec();
                match aliases {
                    Some(aliases) => {
                        let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                        let aliases = aliases.as_object().unwrap();
                        for (k, v) in aliases.into_iter() {
                            out_data.insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                        }
                        out_data.into()
                    }
                    None => data.into(),
                }
            } else {
                let data: Vec<FsdValue> = [
                    buffer.read_f32::<LittleEndian>()?.into(),
                    buffer.read_f32::<LittleEndian>()?.into(),
                ]
                .to_vec();
                match aliases {
                    Some(aliases) => {
                        let mut out_data = std::collections::HashMap::<String, FsdValue>::new();
                        let aliases = aliases.as_object().unwrap();
                        for (k, v) in aliases.into_iter() {
                            out_data.insert(k.clone(), data[v.as_u64().unwrap() as usize].clone());
                        }
                        out_data.into()
                    }
                    None => data.into(),
                }
            }),
            _ => Err(FsdDecodeError::UnsupportedType(s_type.to_string())),
        }
        //
    }
}

fn main() -> Result<(), FsdDecodeError> {
    simple_logger::init_with_level(log::Level::Info).unwrap();

    let matches = App::new("FSD to JSON")
        .version("1.0")
        .author("Alexander Guettler <alexander@guettler.io>")
        .arg(
            Arg::with_name("OUT")
                .short('o')
                .about("Output directory for the resulting .json file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("INPUT")
                .about("The FSD file to be converted to json")
                .required(true)
                .takes_value(true)
                .index(1),
        )
        .get_matches();

    let input_file = matches.value_of("INPUT").unwrap();
    info!("Processing {}", input_file);
    let mut file = std::fs::File::open(input_file)?;

    let mut full_buffer = Vec::new();
    file.read_to_end(&mut full_buffer)?;
    let mut reader = std::io::Cursor::new(&full_buffer);

    trace!("Read Schema");
    let schema = {
        let schema_size = reader.read_u32::<LittleEndian>()?;
        let mut buffer = vec![0; schema_size as usize];
        reader.read_exact(&mut buffer)?;
        let schema = pickle_to_json(&serde_pickle::from_slice(&buffer).unwrap());
        schema
    };
    trace!("{}", schema);

    let _is_sub_object_at_index = schema["valueTypes"]
        .get("buildIndex")
        .map_or(false, |x| x.as_bool().unwrap_or(false))
        && (schema["valueTypes"]["type"] == "dict");

    let offset_to_data = reader.seek(std::io::SeekFrom::Current(0))?;
    let n = FsdValue::from_buffer(&mut reader, offset_to_data, &schema)?;

    let out_file = format!(
        "{}.json",
        Path::new(input_file).file_stem().unwrap().to_str().unwrap()
    );
    let out_file = match matches.value_of("OUT") {
        Some(v) => std::path::Path::new(v),
        None => std::path::Path::new(""),
    }
    .join(out_file);

    let out_file_schema = format!(
        "{}.schema.json",
        Path::new(input_file).file_stem().unwrap().to_str().unwrap()
    );
    let out_file_schema = match matches.value_of("OUT") {
        Some(v) => std::path::Path::new(v),
        None => std::path::Path::new(""),
    }
    .join(out_file_schema);
    let v: serde_json::Value = n.into();
    let vs: serde_json::Value = schema.into();

    std::fs::write(out_file, &serde_json::to_string_pretty(&v).unwrap())?;
    std::fs::write(out_file_schema, &serde_json::to_string_pretty(&vs).unwrap())?;

    Ok(())
}
