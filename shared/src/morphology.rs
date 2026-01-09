#[derive(Clone, Debug)]
pub struct Segment {
    pub length: f32,
    pub mass: f32,
    pub joints: Vec<Joint>,
}

#[derive(Clone, Debug)]
pub struct Joint {
    pub angle_min: f32,
    pub angle_max: f32,
    pub max_torque: f32,
    pub child: Segment,
}

#[derive(Clone, Debug)]
pub struct Morphology {
    pub root: Segment,
}

impl Morphology {
    pub fn humanoid() -> Self {
        Self {
            root: Segment {
                length: 0.2, // Head
                mass: 5.0,
                joints: vec![
                    // Torso
                    Joint {
                        angle_min: -0.1, // Slight lean back
                        angle_max: 0.8,  // Lean forward into the walk
                        max_torque: 150.0,
                        child: Segment {
                            length: 0.6,
                            mass: 35.0,
                            joints: vec![
                                // Left Leg
                                Joint {
                                    angle_min: -0.7, // Swing back
                                    angle_max: 1.4,  // Swing forward (right)
                                    max_torque: 120.0,
                                    child: Segment {
                                        length: 0.4,
                                        mass: 10.0,
                                        joints: vec![Joint {
                                            angle_min: -2.3, // Knee bend (backwards)
                                            angle_max: 0.0,  // Straight
                                            max_torque: 80.0,
                                            child: Segment {
                                                length: 0.4,
                                                mass: 3.5,
                                                joints: vec![],
                                            },
                                        }],
                                    },
                                },
                                // Right Leg
                                Joint {
                                    angle_min: -0.7,
                                    angle_max: 1.4,
                                    max_torque: 120.0,
                                    child: Segment {
                                        length: 0.4,
                                        mass: 10.0,
                                        joints: vec![Joint {
                                            angle_min: -2.3,
                                            angle_max: 0.0,
                                            max_torque: 80.0,
                                            child: Segment {
                                                length: 0.4,
                                                mass: 3.5,
                                                joints: vec![],
                                            },
                                        }],
                                    },
                                },
                            ],
                        },
                    },
                    // Left Arm
                    Joint {
                        angle_min: -1.2,
                        angle_max: 1.2,
                        max_torque: 20.0,
                        child: Segment {
                            length: 0.3,
                            mass: 2.5,
                            joints: vec![Joint {
                                angle_min: 0.0,
                                angle_max: 2.2, // Elbow bend forward
                                max_torque: 10.0,
                                child: Segment { length: 0.3, mass: 1.5, joints: vec![] },
                            }],
                        },
                    },
                    // Right Arm
                    Joint {
                        angle_min: -1.2,
                        angle_max: 1.2,
                        max_torque: 20.0,
                        child: Segment {
                            length: 0.3,
                            mass: 2.5,
                            joints: vec![Joint {
                                angle_min: 0.0,
                                angle_max: 2.2,
                                max_torque: 10.0,
                                child: Segment { length: 0.3, mass: 1.5, joints: vec![] },
                            }],
                        },
                    },
                ],
            },
        }
    }
}
