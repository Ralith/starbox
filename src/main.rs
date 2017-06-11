extern crate openexr as exr;
extern crate clap;
#[macro_use]
extern crate error_chain;
extern crate nalgebra as na;
extern crate rand;
extern crate half;

error_chain! {}

use std::fs::File;
use std::path::Path;
use std::f64;

use clap::{Arg, App};
use rand::{Rng, Rand};
use rand::distributions::{IndependentSample, Normal, Exp};
use rand::distributions::normal::StandardNormal;
use half::f16;

quick_main!(run);

fn run() -> Result<()> {
    let args = App::new("starbox")
        .version("0.1")
        .author("Benjamin Saunders <ben.e.saunders@gmail.com>")
        .about("Generates starboxes")
        .arg(Arg::with_name("FILE")
             .help("File to write")
             .required(true))
        .arg(Arg::with_name("resolution")
             .short("r")
             .help("Cubemap edge length")
             .takes_value(true)
             .default_value("2048"))
        .arg(Arg::with_name("number")
             .short("n")
             .help("Number of stars, in thousands")
             .takes_value(true)
             .default_value("500"))
        .get_matches();
    let res = args.value_of("resolution").unwrap().parse().chain_err(|| "failed to parse resolution")?;
    let number: usize = args.value_of("number").unwrap().parse().chain_err(|| "failed to parse number of stars")?;
    let number = 1000 * number;
    println!("uncompressed size: {} MiB", res * res * 6 * 2 * 2 / (1024 * 1024));
    let path = Path::new(args.value_of_os("FILE").unwrap());
    let mut out = File::create(path).chain_err(|| "failed to open output file")?;
    let mut out = exr::ScanlineOutputFile::new(
        &mut out,
        exr::Header::new()
            .set_resolution(res, 6*res)
            .set_envmap(Some(exr::Envmap::Cube))
            .add_channel("Y", exr::PixelType::HALF)
            .add_channel("T", exr::PixelType::HALF))
        .chain_err(|| "failed to initialize encoder")?;

    let zero = f16::from_f32(0.0);
    let mut pixel_data: Vec<(f16, f16)> = vec![(zero, zero); (res * 6 * res) as usize];
    let mut rng = rand::weak_rng();
    let galaxy = Galaxy::rand(&mut rng);
    let viewer = galaxy.star(&mut rng).position;
    for _ in 0..number {
        let star = galaxy.star(&mut rng);
        let vector = star.position - viewer;
        let (face, pos) = project(res, vector);
        let out = &mut pixel_data[(pos.0 + res * (pos.1 + res * face as u32)) as usize];
        let old_irradiance: f32 = out.0.into();
        let old_temp: f32 = out.1.into();

        const SOLAR_LUMINOSITY: f64 = 3.828e26;
        const GALAXY_RADIUS: f64 = 1e21;
        // Conversion from solar luminances per galaxy radius^2 to attowatts/m^2
        const SCALING_FACTOR: f32 = (1e18 * (SOLAR_LUMINOSITY / (GALAXY_RADIUS * GALAXY_RADIUS))) as f32;

        let irradiance = (SCALING_FACTOR * star.intensity / na::norm(&vector).powi(2)).min(half::consts::MAX.into());

        if old_irradiance + irradiance > 0.0 {
            *out = (f16::from_f32(old_irradiance + irradiance),
                    f16::from_f32((old_temp * old_irradiance + star.temperature * irradiance)
                                  / (old_irradiance + irradiance)));
        }
    }

    {
        let mut fb = exr::FrameBuffer::new(res, 6*res);
        fb.insert_channels(&["Y", "T"], &pixel_data);
        out.write_pixels(&fb).chain_err(|| "failed to output data")?;
    }

    Ok(())
}

struct Galaxy {
    rotation: na::UnitQuaternion<f32>,
}

impl Galaxy {
    fn star<R: Rng>(&self, rng: &mut R) -> Star {
        let y = Normal::new(0.0, 0.25);
        let xz = Normal::new(0.0, 1.0);
        let pos = na::Point3::new(xz.ind_sample(rng) as f32, y.ind_sample(rng) as f32, xz.ind_sample(rng) as f32);
        let pos = self.rotation * pos;

        //
        // units below are wrt. sol
        //

        let mass = Exp::new(1.0).ind_sample(rng);

        let radius = 0.43039846 * mass + 0.52963256; // TODO: Fudge

        // Mass-luminosity relation
        // Main-Sequence Effective Temperatures from a Revised Mass-Luminosity Relation Based on Accurate Properties
        // Z. Eker, F. Soydugan, E. Soydugan, S. Bilir, E. Yaz Gokce, I. Steer, M. Tuysuz, T. Senyuz, O. Demircan (2015)
        let luminosity = if mass <= 1.05 {
            4.841 * mass.ln() - 0.026
        } else if mass <= 2.40 {
            4.328 * mass.ln() - 0.002
        } else if mass <= 7.0 {
            3.962 * mass.ln() + 0.120
        } else {
            2.726 * mass.ln() + 1.237
        }.exp();

        let temperature = 5777.0 * (luminosity / radius.powi(2)).powf(0.25);

        Star {
            position: pos,
            intensity: (luminosity / (4.0 * f64::consts::PI)) as f32,
            temperature: temperature as f32,
        }
    }
}

impl Rand for Galaxy {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let StandardNormal(x) = rng.gen();
        let StandardNormal(y) = rng.gen();
        let StandardNormal(z) = rng.gen();
        let StandardNormal(w) = rng.gen();
        let q = na::Unit::new_normalize(na::Quaternion::new(w, x, y, z));
        Galaxy {
            rotation: na::convert(q),
        }
    }
}

struct Star {
    position: na::Point3<f32>,
    temperature: f32,
    /// Radiant intensity
    intensity: f32,
}

enum Face {
    PX = 0,
    NX = 1,
    PY = 2,
    NY = 3,
    PZ = 4,
    NZ = 5,
}

fn project(res: u32, n: na::Vector3<f32>) -> (Face, (u32, u32)) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let face: Face;
    let pos: (f32, f32);
    if ax >= ay && ax >= az {
        if ax == 0.0 {
            return (Face::PX, (0, 0));
        }
        pos = ((n.y / ax + 1.0) / 2.0 * (res - 1) as f32,
               (n.z / ax + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.x > 0.0 {
            Face::PX
        } else {
            Face::NX
        };
    } else if ay >= az {
        pos = ((n.x / ay + 1.0) / 2.0 * (res - 1) as f32,
               (n.z / ay + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.y > 0.0 {
            Face::PY
        } else {
            Face::NY
        };
    } else {
        pos = ((n.x / az + 1.0) / 2.0 * (res - 1) as f32,
               (n.y / az + 1.0) / 2.0 * (res - 1) as f32);
        face = if n.z > 0.0 {
            Face::PZ
        } else {
            Face::NZ
        };
    }
    (face, (pos.0 as u32, pos.1 as u32))
}
