extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use ml_dataflow::data::{
    dataflow::{ExchangeEvenly, SegmentTrainingData},
    serialization::{AbomonableArray, AsView},
    TrainingData,
};
use ml_dataflow::models::decision_tree::classification::impurity::Gini;
use ml_dataflow::models::decision_tree::classification::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely::dataflow::Scope;
use timely_communication::initialize::Configuration;

fn main() {
    Logger::with_env_or_str("ml_dataflow=debug")
        .start()
        .unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
        let x = arr2(&[
            [10., 5., 10., 10., 50.],
            [10., 5., 10., 10., 50.],
            [1., 0., 2., 0., 3.],
            [1., 0., 2., 0., 3.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [5., 5., 5., 5., 5.],
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.],
        ]);

        let y: Array1<usize> = arr1(&[0, 0, 2, 2, 1, 1, 1, 1, 3, 3]);

        let points_per_worker = 500_000;
        let model = StreamingClassificationTree::new(5, points_per_worker, 5, Gini);

        root.dataflow::<u64, _, _>(|root_scope| {
            let training_stream = vec![
                TrainingData {
                    x: x.clone().into(),
                    y: y.clone().into(),
                },
                TrainingData {
                    x: x.clone().into(),
                    y: y.clone().into(),
                },
            ].to_stream(root_scope);

            let trees = root_scope.scoped::<u64, _, _>(|segment_scope| {
                training_stream
                    .enter(segment_scope)
                    .segment_training_data(points_per_worker * segment_scope.peers() as u64)
                    .exchange_evenly()
                    .train(&model)
                    .inspect(|x| println!("Results: {:?}", x))
                    .leave()
            });

            if root_scope.index() == 0 {
                vec![AbomonableArray::from(x.clone())]
            } else {
                vec![]
            }.to_stream(root_scope)
                .predict(&model, trees.broadcast())
                .inspect(|d| println!("{}", d.as_ref().unwrap().view()));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
