use csv::{Reader as CsvReader, ReaderBuilder};
use data::providers::{DataSource, DataSourceSpec, IntSliceIndex};
use itertools::Itertools;
use serde::Deserialize;
use std::convert::{TryFrom, TryInto};
use std::fs::File;
use std::io;
use std::marker::PhantomData;
use timely::Data;
use Result;

#[derive(Abomonation, Debug, Clone)]
pub struct CsvFileProviderSpec {
    path: String,
    options: CsvProviderOptions,
}

impl TryFrom<CsvFileProviderSpec> for CsvProvider<File, CsvFileProviderSpec> {
    type Error = ::failure::Error;

    fn try_from(spec: CsvFileProviderSpec) -> Result<Self> {
        Ok(CsvProvider::from_spec(spec))
    }
}

impl TryInto<CsvReader<File>> for CsvFileProviderSpec {
    type Error = ::failure::Error;

    fn try_into(self) -> Result<CsvReader<File>> {
        Ok(ReaderBuilder::new()
            .has_headers(self.options.has_headers)
            .delimiter(self.options.delimiter)
            .from_path(self.path)?)
    }
}

#[derive(Abomonation, Debug, Clone)]
pub struct CsvStringProviderSpec {
    content: String,
    options: CsvProviderOptions,
}

impl TryInto<CsvReader<io::Cursor<Vec<u8>>>> for CsvStringProviderSpec {
    type Error = ::failure::Error;

    fn try_into(self) -> Result<CsvReader<io::Cursor<Vec<u8>>>> {
        Ok(ReaderBuilder::new()
            .has_headers(self.options.has_headers)
            .delimiter(self.options.delimiter)
            .from_reader(io::Cursor::new(self.content.into_bytes())))
    }
}

impl CsvStringProviderSpec {
    pub fn new(content: String, options: CsvProviderOptions) -> Self {
        CsvStringProviderSpec { content, options }
    }
}

impl TryFrom<CsvStringProviderSpec> for CsvProvider<io::Cursor<Vec<u8>>, CsvStringProviderSpec> {
    type Error = ::failure::Error;

    fn try_from(spec: CsvStringProviderSpec) -> Result<Self> {
        Ok(CsvProvider::from_spec(spec))
    }
}

#[derive(Abomonation, Debug, Clone)]
pub struct CsvProviderOptions {
    pub has_headers: bool,
    pub delimiter: u8,
}

impl Default for CsvProviderOptions {
    fn default() -> Self {
        CsvProviderOptions {
            has_headers: false,
            delimiter: b',',
        }
    }
}

impl<T: Data> DataSourceSpec<Vec<T>> for CsvFileProviderSpec
where
    for<'de> T: Deserialize<'de>,
{
    type Provider = CsvProvider<File, Self>;
}

impl<T: Data> DataSourceSpec<Vec<T>> for CsvStringProviderSpec
where
    for<'de> T: Deserialize<'de>,
{
    type Provider = CsvProvider<io::Cursor<Vec<u8>>, Self>;
}

pub struct CsvProvider<R, B: Clone + TryInto<CsvReader<R>, Error = ::failure::Error>> {
    builder: B,
    phantom_data: PhantomData<R>,
}

impl<R: io::Read, B: Clone + TryInto<CsvReader<R>, Error = ::failure::Error>> CsvProvider<R, B> {
    pub fn from_spec(builder: B) -> Self {
        CsvProvider {
            builder,
            phantom_data: PhantomData,
        }
    }
}

impl<Reader, ReaderBuilder, Row> DataSource<Vec<Row>>
    for CsvProvider<Reader, ReaderBuilder>
where
    Reader: io::Read,
    ReaderBuilder: Clone + TryInto<CsvReader<Reader>, Error = ::failure::Error>,
    for<'de> Row: Deserialize<'de>,
    Row: Data,
{
    fn slice(&mut self, idx: IntSliceIndex<usize>) -> Result<Vec<Row>> {
        let rows = self.builder
            .clone()
            .try_into()?
            .into_deserialize::<Row>()
            .skip(idx.start)
            .take(idx.length)
            .map(|r| r.map_err(::failure::Error::from))
            .collect::<Result<Vec<_>>>();
        Ok(rows?)
    }

    fn all(&mut self) -> Result<Vec<Row>> {
        let rows = self.builder
            .clone()
            .try_into()?
            .into_deserialize::<Row>()
            .map(|r| r.map_err(::failure::Error::from))
            .collect::<Result<Vec<_>>>();
        Ok(rows?)
    }

    fn select(&mut self, indices: &[usize]) -> Result<Vec<Row>> {
        let rows_iter = self.builder
            .clone()
            .try_into()?
            .into_deserialize::<Row>()
            .enumerate();

        let mut rows = Vec::new();
        let mut i: usize = 0;
        for (j, row) in rows_iter {
            if i >= indices.len() {
                break;
            }
            if indices[i] == j {
                rows.push(row?);
                i += 1;
            }
        }

        Ok(rows)
    }

    fn chunk_indices(
        &mut self,
        chunk_length: usize,
    ) -> Result<Box<Iterator<Item = IntSliceIndex<usize>>>> {
        let chunk_indices = self.builder
            .clone()
            .try_into()
            .unwrap()
            .records()
            .chunks(chunk_length)
            .into_iter()
            .enumerate()
            .map(|(chunk_num, chunk)| IntSliceIndex::new(chunk_num * chunk_length, chunk.count()))
            .collect::<Vec<_>>();
        Ok(Box::new(chunk_indices.into_iter()))
    }

    fn count(&mut self) -> Result<usize> {
        Ok(self.builder.clone().try_into().unwrap().records().count())
    }
}

#[cfg(test)]
mod test {
    use data::providers::csv::*;
    use data::providers::{DataSource, IntSliceIndex};
    use std::convert::TryFrom;

    #[derive(Abomonation, Debug, Deserialize, Clone, PartialEq)]
    struct TestStruct {
        col1: String,
        col2: u64,
        col3: f64,
    }

    #[test]
    fn read_and_deserialize() {
        let csv = "string,11,22.0\n\
                   string2,44,55.0";
        let source = CsvStringProviderSpec::new(csv.to_owned(), CsvProviderOptions::default());
        let mut provider = CsvProvider::try_from(source).expect("Create CSV provider");
        let deserialized: Vec<TestStruct> = provider
            .slice(IntSliceIndex::new(0, 10))
            .expect("Fetch slice");
        assert_eq!(
            vec![
                TestStruct {
                    col1: "string".to_owned(),
                    col2: 11,
                    col3: 22.0,
                },
                TestStruct {
                    col1: "string2".to_owned(),
                    col2: 44,
                    col3: 55.0,
                },
            ],
            deserialized
        );
    }

    #[test]
    fn iter_chunks() {
        let mut csv = String::new();
        for _ in 0..100 {
            csv += "string,11,22.0\n";
        }
        let source = CsvStringProviderSpec::new(csv.to_owned(), CsvProviderOptions::default());
        let mut provider = CsvProvider::try_from(source).expect("Create CSV provider");
        let iterator = <DataSource<Vec<TestStruct>>>::chunk_indices(&mut provider, 10)
            .expect("Get iterator");
        assert_eq!(iterator.count(), 10);
    }
}
