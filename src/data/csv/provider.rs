use csv::ReaderBuilder;
use std::convert::TryFrom;
use std::fs::File;
use data::{DataProviderSpec, Result};
use std::io;

#[derive(Abomonation, Debug, Clone)]
pub struct CsvFileProviderSpec {
    path: String,
    options: CsvProviderOptions,
}

#[derive(Abomonation, Debug, Clone)]
pub struct CsvStringProviderSpec {
    content: String,
    options: CsvProviderOptions,
}

impl CsvStringProviderSpec {
    pub fn new(content: String, options: CsvProviderOptions) -> Self {
        CsvStringProviderSpec { content, options }
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

impl DataProviderSpec for CsvFileProviderSpec {
    type Provider = CsvProvider<File>;
}

impl DataProviderSpec for CsvStringProviderSpec {
    type Provider = CsvProvider<io::Cursor<Vec<u8>>>;
}

pub struct CsvProvider<R: io::Read> {
    reader: ::csv::Reader<R>
}

impl<R: io::Read> CsvProvider<R> {
    pub fn into_reader(self) -> ::csv::Reader<R> {
        self.reader
    }
}

impl TryFrom<CsvFileProviderSpec> for CsvProvider<File> {
    type Error = ::failure::Error;

    fn try_from(spec: CsvFileProviderSpec) -> Result<Self> {
        Ok(CsvProvider::from_reader(File::open(spec.path)?, &spec.options))
    }
}

impl TryFrom<CsvStringProviderSpec> for CsvProvider<io::Cursor<Vec<u8>>> {
    type Error = ::failure::Error;

    fn try_from(spec: CsvStringProviderSpec) -> Result<Self> {
        Ok(CsvProvider::from_reader(io::Cursor::new(spec.content.into_bytes()), &spec.options))
    }
}

impl<R: io::Read> CsvProvider<R> {
    pub fn from_reader(reader: R, options: &CsvProviderOptions) -> CsvProvider<R> {
        CsvProvider {
            reader: ReaderBuilder::new()
                .has_headers(options.has_headers)
                .delimiter(options.delimiter)
                .from_reader(reader)
        }
    }
}
