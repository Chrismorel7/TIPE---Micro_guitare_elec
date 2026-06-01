SetFactory("OpenCASCADE");

// 1. Paramètres (12 aimants, 2 bobines, 1 corde)
For i In {0:5}
  Cylinder(10+i) = {-0.025 + i*0.010, -0.009, -0.015, 0, 0, 0.015, 0.0025}; // Sud
  Cylinder(20+i) = {-0.025 + i*0.010,  0.009, -0.015, 0, 0, 0.015, 0.0025}; // Nord
EndFor

Box(30) = {-0.034, -0.017, -0.012, 0.068, 0.016, 0.012}; // Bobine 1
Box(40) = {-0.034,  0.001, -0.012, 0.068, 0.016, 0.012}; // Bobine 2
Cylinder(50) = {0.005, -0.04, 0.004, 0, 0.08, 0, 0.0005}; // Corde
Box(60) = {-0.06, -0.05, -0.04, 0.12, 0.10, 0.08};        // Air

// 2. Fragmentation
BooleanFragments{ Volume{10:15, 20:25, 30, 40, 50, 60}; Delete; }{}

// 3. Identification par boîte spatiale
v_aimants_sud() = Volume In BoundingBox{-0.03, -0.012, -0.016,  0.03, -0.006, 0.001};
v_aimants_nord() = Volume In BoundingBox{-0.03, 0.006, -0.016,  0.03, 0.012, 0.001};
v_corde() = Volume In BoundingBox{-0.01, -0.05, 0.003,  0.01, 0.05, 0.005};

v_bbox_b1() = Volume In BoundingBox{-0.035, -0.018, -0.013,  0.035, 0, 0.001};
v_bobine_1() = v_bbox_b1(); v_bobine_1() -= v_aimants_sud(); 

v_bbox_b2() = Volume In BoundingBox{-0.035, 0, -0.013,  0.035, 0.018, 0.001};
v_bobine_2() = v_bbox_b2(); v_bobine_2() -= v_aimants_nord(); 

v_air() = Volume "*";
v_air() -= v_aimants_sud(); v_air() -= v_aimants_nord();
v_air() -= v_bobine_1(); v_air() -= v_bobine_2(); v_air() -= v_corde();

// 4. LE SECRET : Numérotation explicite (1 à 6)
Physical Volume("Aimants_Sud",  1) = v_aimants_sud();
Physical Volume("Aimants_Nord", 2) = v_aimants_nord();
Physical Volume("Bobine_1",     3) = v_bobine_1();
Physical Volume("Bobine_2",     4) = v_bobine_2();
Physical Volume("Corde",        5) = v_corde();
Physical Volume("Air",          6) = v_air();

Physical Surface("Frontiere_Air", 7) = Boundary{ Volume{v_air()}; };

// 5. Maillage (un peu plus gros pour économiser la RAM sur le gros modèle)
MeshSize { PointsOf{Volume{v_corde()};} } = 0.001;