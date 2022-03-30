#include "face.h"

#include <TopoDS.hxx>
#include <assert.h>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Bnd_OBB.hxx>
#include <BRepBndLib.hxx>
#include <BRepGProp.hxx>
#include <BRepTools.hxx>
#include <GProp_GProps.hxx>
#include <gp_Pln.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Cone.hxx>
#include <gp_Sphere.hxx>
#include <gp_Torus.hxx>

OCCTFace::OCCTFace(const TopoDS_Shape &shape) {
    assert(shape.ShapeType() == TopAbs_FACE);

    _shape = TopoDS::Face(shape);

    if (_shape.IsNull()) {
        _surf = BRepAdaptor_Surface();
        _has_surface = false;
        orientation = TopAbs_FORWARD; // arbitrary choice
    }
    else {
        _surf = BRepAdaptor_Surface(_shape);
        _has_surface = true;
        orientation = _shape.Orientation();
    }


    init_parametric_function();

    init_bb();

    init_nabb();

    init_mass_props();
}

void OCCTFace::init_parametric_function() {
    if (!_has_surface) {
        function = SurfaceFunction::NONE;
        return;
    }

    GeomAbs_SurfaceType face_class = _surf.GetType();

    switch (face_class) {
    case GeomAbs_Plane:
        init_plane();
        break;
    case GeomAbs_Cylinder:
        init_cyl();
        break;
    case GeomAbs_Cone:
        init_cone();
        break;
    case GeomAbs_Sphere:
        init_sphere();
        break;
    case GeomAbs_Torus:
        init_torus();
        break;
    case GeomAbs_BezierSurface:
        function = SurfaceFunction::BSURF;
        break;
    case GeomAbs_BSplineSurface:
        function = SurfaceFunction::BSURF;
        break;
    case GeomAbs_SurfaceOfExtrusion:
        function = SurfaceFunction::SURFACEOFEXTRUSION;
        break;
    case GeomAbs_SurfaceOfRevolution:
        init_spun();
        break;
    case GeomAbs_OffsetSurface:
        function = SurfaceFunction::OFFSETSURFACE;
        break;
    case GeomAbs_OtherSurface:
        function = SurfaceFunction::OTHERSURFACE;
        break;
    }
}

void OCCTFace::init_bb() {
    if (!_has_surface) {
        bounding_box = Eigen::MatrixXd::Zero(2, 3);
    }
    else {
        Bnd_Box box;
        BRepBndLib::Add(_shape, box);

        double xmin, ymin, zmin, xmax, ymax, zmax;
        box.Get(xmin, ymin, zmin, xmax, ymax, zmax);

        bounding_box.resize(2, 3);
        bounding_box <<
            xmin, ymin, zmin,
            xmax, ymax, zmax;
    }
}

void OCCTFace::init_nabb() {
    // Get Non-Aligned (Oriented) Bounding Box
    // Axes are ordered so X is largest, Y is second largest, Z is smallest

    if (!_has_surface) {
        na_bb_center.setZero();
        na_bb_x.setZero();
        na_bb_z.setZero();
        na_bounding_box = Eigen::MatrixXd::Zero(2, 3);
        return;
    }
    else {
        Bnd_OBB nabox;
        BRepBndLib::AddOBB(_shape, nabox);

        gp_XYZ center = nabox.Center();
        gp_XYZ xdirection = nabox.XDirection();
        gp_XYZ zdirection = nabox.ZDirection();
        double xhsize = nabox.XHSize();
        double yhsize = nabox.YHSize();
        double zhsize = nabox.ZHSize();

        na_bb_center <<
            center.X(),
            center.Y(),
            center.Z();
        na_bb_x <<
            xdirection.X(),
            xdirection.Y(),
            xdirection.Z();
        na_bb_z <<
            zdirection.X(),
            zdirection.Y(),
            zdirection.Z();
        na_bounding_box.resize(2, 3);
        na_bounding_box <<
            -xhsize, -yhsize, -zhsize,
            xhsize, yhsize, zhsize;
    }
}

void OCCTFace::init_mass_props() {
    if (!_has_surface) {
        surface_area = 0;
        center_of_gravity.setZero();
        moment_of_inertia = Eigen::MatrixXd::Zero(3, 3);
        circumference = 0;
        return;
    }
    auto m = MassProperties(_shape);
    surface_area = m.amount;
    center_of_gravity = m.c_of_g;
    moment_of_inertia = m.m_of_i;
    circumference = m.periphery;
}

void OCCTFace::init_plane() {
    gp_Pln plane = _surf.Plane();
    function = SurfaceFunction::PLANE;
    parameters.push_back(plane.Location().X());
    parameters.push_back(plane.Location().Y());
    parameters.push_back(plane.Location().Z());
    parameters.push_back(plane.Axis().Direction().X());
    parameters.push_back(plane.Axis().Direction().Y());
    parameters.push_back(plane.Axis().Direction().Z());
}

void OCCTFace::init_cyl() {
    gp_Cylinder cylinder = _surf.Cylinder();
    function = SurfaceFunction::CYLINDER;
    parameters.push_back(cylinder.Location().X());
    parameters.push_back(cylinder.Location().Y());
    parameters.push_back(cylinder.Location().Z());
    parameters.push_back(cylinder.Axis().Direction().X());
    parameters.push_back(cylinder.Axis().Direction().Y());
    parameters.push_back(cylinder.Axis().Direction().Z());
    parameters.push_back(cylinder.Radius());
}

void OCCTFace::init_cone() {
    gp_Cone cone = _surf.Cone();
    function = SurfaceFunction::CONE;
    parameters.push_back(cone.Location().X());              // 0 - origin_x
    parameters.push_back(cone.Location().Y());              // 1 - origin_y
    parameters.push_back(cone.Location().Z());              // 2 - origin_z
    parameters.push_back(cone.Axis().Direction().X());      // 3 - axis_x
    parameters.push_back(cone.Axis().Direction().Y());      // 4 - axis_y
    parameters.push_back(cone.Axis().Direction().Z());      // 5 - axis_z
    parameters.push_back(cone.RefRadius());                 // 6 - radius
    parameters.push_back(cone.SemiAngle());                 // 7 - semi-angle
}

void OCCTFace::init_sphere() {
    gp_Sphere sphere = _surf.Sphere();
    function = SurfaceFunction::SPHERE;
    parameters.push_back(sphere.Location().X());
    parameters.push_back(sphere.Location().Y());
    parameters.push_back(sphere.Location().Z());
    parameters.push_back(sphere.Position().Axis().Direction().X());
    parameters.push_back(sphere.Position().Axis().Direction().Y());
    parameters.push_back(sphere.Position().Axis().Direction().Z());
    parameters.push_back(sphere.Radius());
}

void OCCTFace::init_torus() {
    gp_Torus torus = _surf.Torus();
    function = SurfaceFunction::TORUS;
    parameters.push_back(torus.Location().X());
    parameters.push_back(torus.Location().Y());
    parameters.push_back(torus.Location().Z());
    parameters.push_back(torus.Axis().Direction().X());
    parameters.push_back(torus.Axis().Direction().Y());
    parameters.push_back(torus.Axis().Direction().Z());
    parameters.push_back(torus.MajorRadius());
    parameters.push_back(torus.MinorRadius());
}

void OCCTFace::init_spun() {
    gp_Ax1 axis = _surf.AxeOfRevolution();
    function = SurfaceFunction::SPUN;
    parameters.push_back(axis.Location().X());
    parameters.push_back(axis.Location().Y());
    parameters.push_back(axis.Location().Z());
    parameters.push_back(axis.Direction().X());
    parameters.push_back(axis.Direction().Y());
    parameters.push_back(axis.Direction().Z());
}
std::vector<Inference> OCCTFace::get_inferences()
{
    std::vector<Inference> inferences;
    switch (function) {
    case SurfaceFunction::PLANE:
        add_inferences_plane(inferences);
        break;
    case SurfaceFunction::CYLINDER:
        add_inferences_axial(inferences);
        break;
    case SurfaceFunction::CONE:
        add_inferences_cone(inferences);
        add_inferences_axial(inferences);
        break;
    case SurfaceFunction::SPHERE:
        add_inferences_sphere(inferences);
        break;
    case SurfaceFunction::TORUS:
        add_inferences_axial(inferences);
        break;
    case SurfaceFunction::SPUN:
        add_inferences_axial(inferences);
        break;
    default:
        break;
    }
    return inferences;
}
void OCCTFace::add_inferences_plane(std::vector<Inference>& inferences)
{
    // Centroid of plane, oriented with the _face_ normal
    // (so flipped if the plane geometry is also)
    Inference inf;
    inf.inference_type = InferenceType::CENTROID;
    inf.origin = Eigen::Vector3d(
        center_of_gravity[0],
        center_of_gravity[1],
        center_of_gravity[2]
    );
    inf.z_axis = Eigen::Vector3d(
        parameters[3],
        parameters[4],
        parameters[5]
    );
    if (orientation == TopAbs_REVERSED) {
        inf.z_axis = -inf.z_axis;
    }
    inferences.push_back(inf);
}
void OCCTFace::add_inferences_cone(std::vector<Inference>& inferences)
{
    // Tip of cone, oriented along cone axis
    double radius = parameters[6];
    double semi_angle = parameters[7];
    Inference tip_inf;
    tip_inf.inference_type = InferenceType::POINT;

    double length = radius / tan(semi_angle);
    for (int i = 0; i < 3; ++i) {
        // parmaters[i] is i-th origin coord
        // parameters[i+3] is i-th axis coord
        tip_inf.origin(i) =
            parameters[i] - parameters[i + 3] * length;
        tip_inf.z_axis(i) = parameters[i + 3];
    }
    inferences.push_back(tip_inf);
}
void OCCTFace::add_inferences_sphere(std::vector<Inference>& inferences)
{
    // Center of sphere, oriented with z-axis
    Inference inf;
    inf.inference_type = InferenceType::CENTER;
    inf.origin = Eigen::Vector3d(parameters[0], parameters[1], parameters[2]);
    inf.z_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
    inferences.push_back(inf);
}
void OCCTFace::add_inferences_axial(std::vector<Inference>& inferences)
{
    // Top, Mid-Point, and Bottom of face as projected onto central
    // axis, oriented parallel to the central axis

    // Find the limits of the face parameterization. Used to determine
    // where the extreme and mid-points are
    // Extract the UV box corners
    double min_u, max_u, center_u, min_v, max_v, center_v;
    BRepTools::UVBounds(_shape, min_u, max_u, min_v, max_v);
    center_u = (min_u + max_u) / 2;
    center_v = (min_v + max_v) / 2;

    // Evaluate the surface function at the extreme and center
    // coordinates for projection onto central axis
    gp_Pnt min_pt, max_pt, center_pt;
    _surf.D0(min_u, min_v, min_pt);
    _surf.D0(max_u, max_v, max_pt);
    _surf.D0(center_u, center_v, center_pt);

    Eigen::Vector3d surf_min(min_pt.X(), min_pt.Y(), min_pt.Z());
    Eigen::Vector3d surf_max(max_pt.X(), max_pt.Y(), max_pt.Z());
    Eigen::Vector3d surf_center(center_pt.X(), center_pt.Y(), center_pt.Z());

    auto project_to_line = [](
        const Eigen::Vector3d& point,
        const Eigen::Vector3d& origin,
        const Eigen::Vector3d& axis) {
            Eigen::Vector3d v = point - origin;
            double length = v.dot(axis) / axis.dot(axis);
            return origin + axis * length;
    };

    // Origin and axis x,y,z components are the first 6 parameters
    Eigen::Vector3d origin = Eigen::Vector3d(parameters[0], parameters[1], parameters[2]);
    Eigen::Vector3d axis = Eigen::Vector3d(parameters[3], parameters[4], parameters[5]);

    Eigen::Vector3d min_axis_point = project_to_line(surf_min, origin, axis);
    Eigen::Vector3d max_axis_point = project_to_line(surf_max, origin, axis);
    Eigen::Vector3d center_axis_point = project_to_line(surf_center, origin, axis);

    Inference bottom_axis, top_axis, mid_axis;

    bottom_axis.inference_type = InferenceType::BOTTOM_AXIS_POINT;
    bottom_axis.origin = min_axis_point;
    bottom_axis.z_axis = axis;

    top_axis.inference_type = InferenceType::TOP_AXIS_POINT;
    top_axis.origin = max_axis_point;
    top_axis.z_axis = axis;

    mid_axis.inference_type = InferenceType::MID_AXIS_POINT;
    mid_axis.origin = center_axis_point;
    mid_axis.z_axis = axis;

    inferences.push_back(bottom_axis);
    inferences.push_back(top_axis);
    inferences.push_back(mid_axis);
}

void OCCTFace::sample_points(
    const int num_points,
    const bool sample_normals,
    std::vector<Eigen::MatrixXd>& samples,
    Eigen::MatrixXd& uv_range) {
    // TODO: implement
}


void OCCTFace::random_sample_points(
    const int num_points,
    Eigen::MatrixXd& samples,
    Eigen::MatrixXd& coords,
    Eigen::MatrixXd& uv_range) {
    // TODO: implement
}
