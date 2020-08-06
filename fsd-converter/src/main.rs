use byteorder::{LittleEndian, ReadBytesExt};
use lazy_static::lazy_static;
use log::trace;
use std::fmt;
use std::io::{BufReader, Read, Seek};

struct PickleValue {
    v: serde_pickle::Value,
}

impl fmt::Display for PickleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl PickleValue {
    // NOTE(alexander): This is incredibly stupid and slow
    // lots of copy happening
    // But access to the serde_pickle::Value thing is also very very stupid
    // and I can't figure the best format to use for the schema
    // so for now this will do until we can write some nice structs for it :)
    pub fn get<T>(&self, s: T) -> Option<Self>
    where
        T: ToString,
        T: ,
    {
        let i = serde_pickle::HashableValue::Bytes(s.to_string().as_bytes().to_vec());
        let s = serde_pickle::HashableValue::String(s.to_string());
        match match &self.v {
            serde_pickle::Value::Set(v) => v.get(&i).map(|x| x.clone().into_value()),
            serde_pickle::Value::FrozenSet(v) => v.get(&i).map(|x| x.clone().into_value()),
            serde_pickle::Value::Dict(v) => v.get(&i).map(|x| x.clone()),
            _ => None,
        } {
            Some(v) => Some(v.into()),
            None => match &self.v {
                serde_pickle::Value::Set(v) => v.get(&s).map(|x| x.clone().into_value().into()),
                serde_pickle::Value::FrozenSet(v) => {
                    v.get(&s).map(|x| x.clone().into_value().into())
                }
                serde_pickle::Value::Dict(v) => v.get(&s).map(|x| x.clone().into()),
                _ => None,
            },
        }
    }
}

impl From<serde_pickle::Value> for PickleValue {
    fn from(v: serde_pickle::Value) -> Self {
        Self { v }
    }
}
impl From<&serde_pickle::Value> for PickleValue {
    fn from(v: &serde_pickle::Value) -> Self {
        Self { v: v.clone() }
    }
}

fn pickle_to_json(v: &serde_pickle::Value) -> serde_json::Value {
    match v {
        serde_pickle::Value::Bool(v) => (*v).into(),
        serde_pickle::Value::I64(v) => (*v).into(),
        serde_pickle::Value::Int(v) => panic!("BigInt not supported"),
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

struct FsdFile {}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum FooterKeyType {
    OffsetIntWithSize,
    OffsetLongWithSize,
    OffsetInt,
    OffsetLong,
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
    pub key: u64,
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
        if self.index >= self.footer.size as usize {
            None
        } else {
            let mut reader = std::io::Cursor::new(&self.footer.buffer);

            reader
                .seek(std::io::SeekFrom::Start(
                    self.index as u64 * self.footer.item_size + self.footer.start_offset,
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
            Some(Self::Item { key, offset, size })
        }
    }
}

impl DictFooter {
    pub fn new(buffer: Vec<u8>, schema: serde_json::Value) -> Self {
        let mut reader = BufReader::new(&buffer[..]);
        let size = reader.read_u32::<LittleEndian>().unwrap();
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
                std::mem::size_of::<u32>() + std::mem::size_of::<u32>() + std::mem::size_of::<u32>()
            }
            FooterKeyType::OffsetLong => std::mem::size_of::<u64>() + std::mem::size_of::<u32>(),
            FooterKeyType::OffsetLongWithSize => {
                std::mem::size_of::<u64>() + std::mem::size_of::<u32>() + std::mem::size_of::<u32>()
            }
        } as u64;

        Self {
            schema,
            buffer,
            key_type,
            size,
            item_size,
            start_offset: 4,
        }
    }
}

fn bool_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    buffer.seek(std::io::SeekFrom::Start(offset));
    (buffer.read_u8().unwrap() == 255).into()
}

fn int_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    buffer.seek(std::io::SeekFrom::Start(offset));

    let min = schema.get("min");
    let exclusive_min = schema.get("exclusiveMin");
    if min.is_some() && min.map(|x| x.as_i64().unwrap_or(0)).unwrap_or(0) >= 0
        || exclusive_min.is_some()
            && exclusive_min.map(|x| x.as_i64().unwrap_or(0)).unwrap_or(0) >= -1
    {
        buffer.read_u32::<LittleEndian>().unwrap().into()
    } else {
        buffer.read_i32::<LittleEndian>().unwrap().into()
    }
}

fn long_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    buffer.seek(std::io::SeekFrom::Start(offset));
    buffer.read_u64::<LittleEndian>().unwrap().into()
}

fn float_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    buffer.seek(std::io::SeekFrom::Start(offset));
    if schema
        .get("preci")
        .map(|x| x.as_str().unwrap_or("single"))
        .unwrap_or("single")
        == "double"
    {
        buffer.read_f64::<LittleEndian>().unwrap().into()
    } else {
        buffer.read_f32::<LittleEndian>().unwrap().into()
    }
}

fn as_u32_be(array: &[u8]) -> u32 {
    ((array[0] as u32) << 24)
        | ((array[1] as u32) << 16)
        | ((array[2] as u32) << 8)
        | ((array[3] as u32) << 0)
}

fn as_u32_le(array: &[u8]) -> u32 {
    ((array[0] as u32) << 0)
        | ((array[1] as u32) << 8)
        | ((array[2] as u32) << 16)
        | ((array[3] as u32) << 24)
}

fn object_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let mut output = serde_json::Value::Null;
    let has_size = schema.get("size").is_some();
    let mut variable_data_offset_base = 0u64;
    let mut offset_attributes_lookup_table = std::collections::HashMap::new();
    if !has_size {
        let mut offsetAttributes: Vec<String> = schema["attributesWithVariableOffsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let optional_value_lookups = schema["optionalValueLookups"].as_object().unwrap();
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
                    offsetAttributes.retain(|x| x != k);
                }
            }
        }
        let offsetAttributeArrayStart = offset + end_of_fixed_size_data + 8;
        let offsetAttributeOffsetsTypeSize = 4 * offsetAttributes.len();
        variable_data_offset_base =
            offsetAttributeArrayStart + offsetAttributeOffsetsTypeSize as u64;
        buffer
            .seek(std::io::SeekFrom::Start(offsetAttributeArrayStart))
            .unwrap();
        let mut offsetData = vec![0; offsetAttributeOffsetsTypeSize];
        buffer.read_exact(&mut offsetData);
        let offsetTable: Vec<u32> = offsetData
            .chunks_exact(4)
            .into_iter()
            .map(|a| as_u32_le(a))
            .collect();
        for (k, v) in offsetAttributes.iter().zip(offsetTable.iter()) {
            offset_attributes_lookup_table.insert(k.clone(), *v);
        }

        println!("No SIZE!");
    }

    let attributes = schema["attributes"].as_object().unwrap();
    for (k, attribute_schema) in attributes.iter() {
        //
        if schema["constantAttributeOffsets"].get(k).is_some() {
            output[k] = parse_value_from_buffer(
                &mut buffer,
                offset + schema["constantAttributeOffsets"][k].as_u64().unwrap(),
                (*attribute_schema).clone(),
            );
        } else {
            //
            if !offset_attributes_lookup_table.contains_key(k) {
                if attribute_schema.get("default").is_some() {
                    output[k] = attribute_schema["default"].clone();
                }
            } else {
                println!(
                    "Load {} at {} {}",
                    k,
                    variable_data_offset_base,
                    variable_data_offset_base + offset_attributes_lookup_table[k] as u64
                );
                output[k] = parse_value_from_buffer(
                    &mut buffer,
                    variable_data_offset_base + offset_attributes_lookup_table[k] as u64,
                    (*attribute_schema).clone(),
                );
            }
        }
    }

    output
}

fn string_from_buffer(
    buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    _: &serde_json::Value,
) -> serde_json::Value {
    buffer.seek(std::io::SeekFrom::Start(offset));
    let count = buffer.read_u32::<LittleEndian>().unwrap();
    let mut str_buffer = vec![0; count as usize];
    buffer.read_exact(&mut str_buffer).unwrap();

    std::str::from_utf8(&str_buffer).unwrap().to_string().into()
}

fn FixedSizeListRepresentation(
    buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let fixedLength = false;
    buffer.seek(std::io::SeekFrom::Start(offset));
    let count = buffer.read_u32::<LittleEndian>().unwrap();
    let itemSize = schema["size"].as_u64().unwrap();

    let countOffset = if fixedLength { 0 } else { 4 };
    let mut output = Vec::new();
    for i in 0..count {
        //
        let totalOffset = offset + countOffset + itemSize * i as u64;
        output.push(parse_value_from_buffer(buffer, totalOffset, schema.clone()));
    }
    serde_json::Value::Array(output)
}

fn VariableSizedListRepresentation(
    buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let fixedLength = false;
    buffer.seek(std::io::SeekFrom::Start(offset));
    let count = buffer.read_u32::<LittleEndian>().unwrap();

    let countOffset = if fixedLength { 0 } else { 4 };
    let mut output = Vec::new();
    for i in 0..count {
        //
        buffer.seek(std::io::SeekFrom::Start(
            offset + countOffset + 4 * i as u64,
        ));
        let n = buffer.read_u32::<LittleEndian>().unwrap();
        let dataOffsetFromObjectStart = offset + n as u64;
        output.push(parse_value_from_buffer(
            buffer,
            dataOffsetFromObjectStart,
            schema.clone(),
        ));
    }
    serde_json::Value::Array(output)
}

fn list_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value {
    let is_fixed_size = schema.get("fixedItemSize").is_some();
    if is_fixed_size {
        FixedSizeListRepresentation(&mut buffer, offset, &schema["itemTypes"])
    } else {
        VariableSizedListRepresentation(&mut buffer, offset, &schema["itemTypes"])
    }
}

type LoaderFn = fn(
    buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: &serde_json::Value,
) -> serde_json::Value;

lazy_static! {
    static ref loaders: std::collections::HashMap<String, LoaderFn> = {
        let mut m: std::collections::HashMap<String, LoaderFn> = std::collections::HashMap::new();
        m.insert("float".to_string(), float_from_buffer);
        m.insert("bool".to_string(), bool_from_buffer);
        m.insert("int".to_string(), int_from_buffer);
        m.insert("long".to_string(), long_from_buffer);
        m.insert("typeID".to_string(), int_from_buffer);
        m.insert("localizationID".to_string(), int_from_buffer);
        m.insert("object".to_string(), object_from_buffer);
        m.insert("string".to_string(), string_from_buffer);
        m.insert("list".to_string(), list_from_buffer);
        m.insert("npcTag".to_string(), int_from_buffer);
        m.insert("deploymentType".to_string(), int_from_buffer);
        m.insert("npcEnemyFleetTypeID".to_string(), int_from_buffer);
        m.insert("groupBehaviorTreeID".to_string(), int_from_buffer);
        m.insert("npcCorporationID".to_string(), int_from_buffer);
        m.insert("spawnTableID".to_string(), int_from_buffer);
        m.insert("npcFleetCounterTableID".to_string(), int_from_buffer);
        m.insert("dungeonID".to_string(), int_from_buffer);
        m.insert("typeListID".to_string(), int_from_buffer);
        m.insert("npcFleetTypeID".to_string(), int_from_buffer);
        m
    };
}

fn parse_value_from_buffer(
    mut buffer: &mut std::io::Cursor<&Vec<u8>>,
    offset: u64,
    schema: serde_json::Value,
) -> serde_json::Value {
    let s_type = schema["type"].as_str().unwrap().to_string();
    (loaders.get(&s_type).unwrap())(&mut buffer, offset, &schema)
}

fn main() -> Result<(), std::io::Error> {
    simple_logger::init_with_level(log::Level::Trace).unwrap();
    let mut file = std::fs::File::open("0.sd")?;

    let mut full_buffer = Vec::new();
    file.read_to_end(&mut full_buffer)?;
    let mut reader = std::io::Cursor::new(&full_buffer); // BufReader::new(&file);

    trace!("Read Schema");
    let schema_size = reader.read_u32::<LittleEndian>()?;
    let mut buffer = vec![0; schema_size as usize];
    reader.read_exact(&mut buffer)?;
    let schema = pickle_to_json(&serde_pickle::from_slice(&buffer).unwrap());
    trace!("Read Footer");
    let size_of_data = reader.read_u32::<LittleEndian>()?;
    println!("{}", size_of_data);
    let offset_to_data = reader.seek(std::io::SeekFrom::Current(0))?;
    trace!("Jump to end of data");
    reader.seek(std::io::SeekFrom::Current((size_of_data - 4) as i64))?; // Jump to the end of the data, which is where the footer size is
    trace!("Read footer size");
    let size_of_footer = reader.read_u32::<LittleEndian>()?;
    trace!("Jump to footer start");
    reader.seek(std::io::SeekFrom::Current(-1 * (4 + size_of_footer as i64)))?; // // Jump to the start of the footer // offset_to_data + size_of_data as u64 - size_of_footer as u64,

    let mut buffer = vec![0; size_of_footer as usize];
    reader.read_exact(&mut buffer)?;

    let footer = DictFooter::new(buffer, schema.clone());
    let value_schema = schema["valueTypes"].clone();
    for n in footer.into_iter() {
        //
        reader
            .seek(std::io::SeekFrom::Start(offset_to_data + n.offset as u64))
            .unwrap();
        match n.size {
            Some(size) => {
                println!("{}", size);
                let mut item_buffer = vec![0; size as usize];
                let off = reader.seek(std::io::SeekFrom::Current(0))?;
                println!("Reading from {}", off);
                reader.read_exact(&mut item_buffer)?;
                let n = parse_value_from_buffer(
                    &mut std::io::Cursor::new(&item_buffer),
                    0,
                    schema["valueTypes"].clone(),
                );
                println!("Read ITEM {}", n);
            }
            None => {}
        };
    }

    // Make sure we are actually at end of the file now!
    // assert!(reader.read_u32::<LittleEndian>().is_err());
    reader.seek(std::io::SeekFrom::Start(offset_to_data))?;

    // Everthing here essentially operates of offset_to_data

    let isSubObjectAnIndex = schema["valueTypes"]
        .get("buildIndex")
        .map_or(false, |x| x.as_bool().unwrap_or(false))
        & (schema["valueTypes"]["type"] == "dict");
    println!("{}", size_of_data);
    Ok(())
}
