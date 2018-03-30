use data::{
    csv::provider::*, generic_ops::FetchSlice,
    IntoDataOperator,
    Result,
};
use serde::Deserialize;
use std::io;
use timely::Data;

impl<Reader, R> IntoDataOperator<R, CsvProvider<Reader>> for FetchSlice
    where Reader: io::Read, R: Data, for<'de> R: Deserialize<'de>
{
    fn fetch_vec(&self, provider: CsvProvider<Reader>) -> Result<Vec<R>> {
        provider.into_reader()
            .into_deserialize::<R>()
            .skip(self.start_index)
            .take(self.length)
            .map(|r| r.map_err(::failure::Error::from))
            .collect::<Result<Vec<_>>>()
    }

    fn fetch_into_iter(&self, provider: CsvProvider<Reader>) -> Result<Box<Iterator<Item=R>>> {
        Ok(Box::new(
            self.fetch_vec(provider)?.into_iter()
        ))
    }
}
