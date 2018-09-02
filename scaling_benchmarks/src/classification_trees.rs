extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate timely;
extern crate timely_communication;
extern crate toml;
#[macro_use]
extern crate serde_derive;
extern crate serde;

use flexi_logger::Logger;
use ml_dataflow::models::decision_tree::classification::impurity::Gini;
use timely::dataflow::Stream;

use self::config::ClassificationTrees;
use ml_dataflow::data::dataflow::random::{params::*, RandClassificationTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{IncorrectRatio, MeasurePredictionError},
    TrainingData,
};
use ml_dataflow::models::decision_tree::classification::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

mod config;

fn main() {
    let config = self::config::read().classification_trees;
    Logger::with_env_or_str("ml_dataflow=info").start().unwrap();

    println!("Run for bins {:?}", config.bins);
    for bins in config.bins.clone() {
        run(config.clone(), bins, 8)
    }

    println!("Run for threads {:?}", config.threads);
    for threads in config.threads.clone() {
        println!("Run with {} threads", threads);
        run(config.clone(), 3, threads)
    }
}

fn run(config: ClassificationTrees, bins: usize, threads: usize) {
    let samples_per_thread = config.samples / threads;
    ::timely::execute(Configuration::Process(threads), move |root| {
        let config = config.clone();

        let rand_source = RandClassificationTrainingSource::default()
            .samples(samples_per_thread, 1, 1)
            .x_distributions(arr2(&[
                [
                    NormalParams::new(1., 1.),
                    NormalParams::new(2., 1.),
                    NormalParams::new(3., 1.),
                    NormalParams::new(-1., 1.),
                    NormalParams::new(5., 1.),
                ],
                [
                    NormalParams::new(3., 1.),
                    NormalParams::new(0., 1.),
                    NormalParams::new(3., 1.),
                    NormalParams::new(-1., 1.),
                    NormalParams::new(4., 1.),
                ],
                [
                    NormalParams::new(5., 1.),
                    NormalParams::new(2., 1.),
                    NormalParams::new(1., 1.),
                    NormalParams::new(1., 1.),
                    NormalParams::new(7., 1.),
                ],
                [
                    NormalParams::new(0., 1.),
                    NormalParams::new(4., 1.),
                    NormalParams::new(0., 1.),
                    NormalParams::new(0., 1.),
                    NormalParams::new(7., 1.),
                ],
                [
                    NormalParams::new(10., 1.),
                    NormalParams::new(2., 1.),
                    NormalParams::new(-3., 1.),
                    NormalParams::new(6., 1.),
                    NormalParams::new(3., 1.),
                ],
            ]))
            .y_distributions(arr1(&[
                DummyDistribution(0),
                DummyDistribution(1),
                DummyDistribution(2),
                DummyDistribution(3),
                DummyDistribution(4),
            ]));

        let model = StreamingClassificationTree::new(
            config.levels as u64,
            samples_per_thread as u64,
            bins,
            Gini,
        );

        root.dataflow::<u64, _, _>(move |root_scope| {
            let training_stream: Stream<_, TrainingData<f64, i64>> = rand_source
                .clone()
                .samples(samples_per_thread as usize, 1, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope);

            let trees = training_stream.train(&model);

            let predict_data = rand_source.clone().samples(5000, 1, 1).to_stream(
                Summary::Local(1),
                RootTimestamp::new(1),
                root_scope,
            );

            predict_data
                .map(|t_d| t_d.x)
                .predict(&model, trees.broadcast())
                .map(|res| res.expect("prediction"))
                .prediction_error(&predict_data.map(|t_d| t_d.y), IncorrectRatio)
                .inspect(move |incorr| println!("{} Bins, Incorrect: {}", bins, incorr));
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
