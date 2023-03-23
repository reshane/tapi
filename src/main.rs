use tch::vision::{
    imagenet,
    resnet,
};
use tch::nn::ModuleT;
//use rocket::serde::{Deserialize, json::Json};
use rocket::{Rocket, Build};

#[macro_use] extern crate rocket;

#[get("/")]
fn hello() -> String {
    let image_file = "tiger_maybe.jpg";
    let image = imagenet::load_image_and_resize224(image_file).unwrap();
    let model_file = "resnet18.ot";
    let weights = std::path::Path::new(model_file);
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net: Box<dyn ModuleT> = Box::new(resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT));
    vs.load(weights).unwrap();
    let output = net.forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float);
    format!("{:?}", imagenet::top(&output, 1))
}

#[rocket::main]
async fn main() -> Result<(), rocket::Error> {
    let _rocket = rocket::build()
        .mount("/", routes![hello])
        .launch()
        .await?;
    Ok(())
}
