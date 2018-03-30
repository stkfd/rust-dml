mod provider;
mod query;

#[cfg(test)]
mod test {
    use super::provider::*;
    use super::query::*;
    use data::IntoDataOperator;
    use std::convert::TryFrom;
    use abomonation::Abomonation;

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
        let provider = CsvProvider::try_from(source).expect("Create CSV provider");
        let deserialized: Vec<TestStruct> = FetchSlice::new(0, 10).execute(provider).expect("Execute slice query");
        assert_eq!(vec![TestStruct { col1: "string".to_owned(), col2: 11, col3: 22.0 }, TestStruct { col1: "string2".to_owned(), col2: 44, col3: 55.0 }], deserialized);
    }
}
