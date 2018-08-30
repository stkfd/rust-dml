use data::TrainingData;
use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::Num;
use std::fmt::Debug;
use std::str::FromStr;
use timely::dataflow::operators::generic::source;
use timely::dataflow::{Scope, Stream};
use timely::Data;

pub trait CsvTrainingDataSource<S: Scope, T, L> {
    fn training_data_from_csv(
        &self,
        path: String,
        chunk_size: usize,
    ) -> Stream<S, TrainingData<T, L>>;
}

impl<S: Scope, T, L> CsvTrainingDataSource<S, T, L> for S
where
    T: Data + Num + FromStr,
    <L as FromStr>::Err: Debug,
    L: Data + Num + FromStr,
    <T as FromStr>::Err: Debug,
{
    fn training_data_from_csv(
        &self,
        path: String,
        chunk_size: usize,
    ) -> Stream<S, TrainingData<T, L>> {
        source(self, "CsvSource", move |default_cap| {
            let mut cap = Some(default_cap);
            move |output| {
                if let Some(cap) = cap.take() {
                    let mut session = output.session(&cap);
                    let mut rdr = ::csv::Reader::from_path(path.clone()).expect("open csv training data");
                    for chunk in &rdr.into_records().chunks(chunk_size) {
                        let mut x_opt = None;
                        let mut y = Array1::<L>::zeros(chunk_size);
                        let mut row = 0;
                        for record in chunk {
                            let record = record.expect("Read line");
                            if x_opt.is_none() {
                                x_opt = Some(Array2::<T>::zeros((chunk_size, record.len() - 1)));
                            }
                            let mut x_row = x_opt.as_mut().unwrap().row_mut(row);

                            let mut columns = record.iter().collect::<Vec<_>>();
                            for (x, csv_item) in
                                x_row.iter_mut().zip(columns[0..(columns.len() - 1)].iter())
                            {
                                *x = csv_item.parse().expect("parse input value");
                            }
                            y[row] = columns[columns.len() - 1]
                                .parse()
                                .expect("parse output value");
                            row += 1;
                        }
                        println!("chunk_size: {} row: {}", chunk_size, row);
                        let mut x = x_opt.unwrap();
                        if row < chunk_size {
                            x.slice_axis_inplace(Axis(0), (0..row).into());
                            y.slice_axis_inplace(Axis(0), (0..row).into());
                        }
                        session.give(TrainingData { x: x.into(), y: y.into()})
                    }
                }
            }
        })
    }
}
