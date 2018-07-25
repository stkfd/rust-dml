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
use ml_dataflow::models::decision_tree::regression::StreamingRegressionTree;
use ml_dataflow::models::gradient_boost::GradientBoostingRegression;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely::dataflow::Scope;
use timely_communication::initialize::Configuration;

fn main() {
    Logger::with_env_or_str("ml_dataflow=info,ml_dataflow::models::gradient_boost=debug")
        .start()
        .unwrap();
    ::timely::execute(Configuration::Process(2), move |root| {
        let x = arr2(&[[0], [0], [1], [1], [2], [2], [3], [3]]);

        let y: Array1<f64> = arr1(&[10., 10., 12., 12., 14., 14., 16., 16.]);

        let points_per_worker = 500_000;
        let regression_tree_model = StreamingRegressionTree::new(1, points_per_worker, 5, 1.0);
        let gradient_boosting_model = GradientBoostingRegression::new(3, regression_tree_model);

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

            let boost_chain = root_scope.scoped::<u64, _, _>(|segment_scope| {
                training_stream
                    .enter(segment_scope)
                    .segment_training_data(points_per_worker * segment_scope.peers() as u64)
                    .exchange_evenly()
                    .train_meta(&gradient_boosting_model)
                    .inspect(|x| println!("Results: {:?}", x))
                    .leave()
            });

            let in_stream = vec![AbomonableArray::from(x.clone())].to_stream(root_scope);

            in_stream
                .predict(&gradient_boosting_model, boost_chain.broadcast())
                .inspect(|d| println!("{}", d.as_ref().unwrap().view()));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
