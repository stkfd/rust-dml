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
use ml_dataflow::models::kmeans::initializers::RandomSample;
use ml_dataflow::models::kmeans::ConvergenceCriteria;
use timely::dataflow::Stream;

use self::config::Kmeans as KmeansConfig;
use ml_dataflow::data::dataflow::random::{params::*, RandClassificationTrainingSource};
use ml_dataflow::data::serialization::AsView;
use ml_dataflow::data::TrainingData;
use ml_dataflow::models::kmeans::Kmeans;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use std::time::Instant;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;
mod config;

fn main() {
    let config = self::config::read().kmeans;
    Logger::with_env_or_str("ml_dataflow=debug").start().unwrap();

    println!("Run for threads {:?}", config.threads);
    for threads in config.threads.clone() {
        let inst = Instant::now();
        println!("Run with {} threads", threads);
        run(config.clone(), threads);
        println!("took {:?}", inst.elapsed());
    }
}

fn run(config: KmeansConfig, threads: usize) {
    let samples_per_thread = config.samples / threads;
    ::timely::execute(Configuration::Process(threads), move |root| {
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

        let end_criteria = <ConvergenceCriteria<f64>>::default()
            .limit_iterations(5);
        let model = Kmeans::<_, RandomSample>::new(5, 5, end_criteria);

        root.dataflow::<u64, _, _>(move |root_scope| {
            let training_stream: Stream<_, TrainingData<f64, i64>> = rand_source
                .clone()
                .samples(samples_per_thread as usize, 1, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope);

            let result = training_stream.map(|td| td.x).train(&model);
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
