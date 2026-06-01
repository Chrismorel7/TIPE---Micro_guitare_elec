SetFactory("OpenCASCADE");

// 1. Paramètres
For i In {0:5}
  Cylinder(10+i) = {-0.025 + i*0.010, 0, -0.015, 0, 0, 0.015, 0.0025}; // 6 aimants alignés
EndFor

Box(20) = {-0.034, -0.008, -0.012, 0.068, 0.016, 0.012}; // Une seule bobine globale
Cylinder(30) = {0.005, -0.04, 0.004, 0, 0.08, 0, 0.0005}; // Corde
Box(40) = {-0.06, -0.05, -0.04, 0.12, 0.10, 0.08};        // Air

// 2. Fragmentation
BooleanFragments{ Volume{10:15, 20, 30, 40}; Delete; }{}

// 3. Identification spatiale
v_aimants() = Volume In BoundingBox{-0.03, -0.004, -0.016,  0.03, 0.004, 0.001};
v_corde() = Volume In BoundingBox{-0.01, -0.05, 0.003,  0.01, 0.05, 0.005};

v_bbox_b1() = Volume In BoundingBox{-0.035, -0.009, -0.013,  0.035, 0.009, 0.001};
v_bobine() = v_bbox_b1(); v_bobine() -= v_aimants(); 

v_air() = Volume "*";
v_air() -= v_aimants(); v_air() -= v_bobine(); v_air() -= v_corde();

// 4. Fixation des IDs pour ParaView
Physical Volume("Aimants", 1) = v_aimants();
Physical Volume("Bobine",  2) = v_bobine();
Physical Volume("Corde",   3) = v_corde();
Physical Volume("Air",     4) = v_air();

Physical Surface("Frontiere_Air", 5) = Boundary{ Volume{v_air()}; };

// 5. Maillage
MeshSize { PointsOf{Volume{v_corde()};} } = 0.001;