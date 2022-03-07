#include "body.h"

#include <set>
#include <map>
#include <unordered_map>
#include <vector>

#include <STEPControl_Reader.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopExp.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <Poly_Triangulation.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <TopLoc_Location.hxx>
#include <gp_Pnt.hxx>
#include <Interface_Static.hxx>

std::vector<std::shared_ptr<Body>> read_step(std::string path) {
    std::vector<std::shared_ptr<Body>> parts_vec;

	STEPControl_Reader reader;
    Interface_Static::SetCVal("xstep.cascade.unit", "M");
	IFSelect_ReturnStatus ret = reader.ReadFile(path.c_str());

    if (ret == IFSelect_RetDone) {
        reader.TransferRoots();
        TopoDS_Shape shape = reader.OneShape();
		
        parts_vec.emplace_back(new OCCTBody(shape));
    }

    return parts_vec;
}

OCCTBody::OCCTBody(const TopoDS_Shape& shape) {
    _shape = shape;

    // Create _shape_to_idx to associate entities with logical IDs
    TopTools_IndexedMapOfShape shape_map;
    TopExp::MapShapes(_shape, shape_map);

    int i = 0;
    for (auto iterator = shape_map.cbegin(); iterator != shape_map.cend(); iterator++) {
        TopoDS_Shape subshape = *iterator;
        _shape_to_idx[subshape] = i;
        i++;
    }

    _valid = true;
}

BREPTopology OCCTBody::GetTopology() {
    // TODO: handle duplicate edges that only differ in orientation

    BREPTopology topology;

    std::map<int, int> cat_idx;

    TopTools_IndexedDataMapOfShapeListOfShape edge_face_map;
    TopExp::MapShapesAndAncestors(_shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map);

    TopTools_IndexedDataMapOfShapeListOfShape loop_face_map;
    TopExp::MapShapesAndAncestors(_shape, TopAbs_WIRE, TopAbs_FACE, loop_face_map);

    // Store faces, loops, edges, vertices
    int i = 0;
    for (const auto& subshape_idx_pair : _shape_to_idx) {
        TopoDS_Shape subshape = subshape_idx_pair.first;
        int idx = subshape_idx_pair.second;

        switch (subshape.ShapeType()) {
        case TopAbs_FACE:
            topology.pk_to_class[idx] = PK_CLASS_face;
            cat_idx[i] = topology.faces.size();
            topology.faces.emplace_back(new OCCTFace(subshape));
            break;
        case TopAbs_WIRE:
            topology.pk_to_class[idx] = PK_CLASS_loop;
            cat_idx[i] = topology.loops.size();
            topology.loops.emplace_back(new OCCTLoop(subshape, loop_face_map.FindFromKey(subshape)));
            break;
        case TopAbs_EDGE:
            topology.pk_to_class[idx] = PK_CLASS_edge;
            cat_idx[i] = topology.edges.size();
            topology.edges.emplace_back(new OCCTEdge(subshape, edge_face_map.FindFromKey(subshape)));
            break;
        case TopAbs_VERTEX:
            topology.pk_to_class[idx] = PK_CLASS_vertex;
            cat_idx[i] = topology.vertices.size();
            topology.vertices.emplace_back(new OCCTVertex(subshape));
            break;
        default:
            break;
        }

        // Update map from pk index to index with each entity type
        topology.pk_to_idx[idx] = cat_idx[i];

        i++;
    }

    // Maps for finding face-face edges
    std::map<int, int> loop_to_face;
    std::map<int, std::vector<int> > edge_to_loop;

    std::map<int, std::vector<int> > face_to_loops;
    std::map<int, std::vector<int> > loop_to_edges;
    std::map<int, std::vector<int> > loop_to_vertex;
    std::map<int, std::vector<int> > edge_to_vertices;
    
    TopTools_IndexedMapOfShape subshape_map;

    // Fill topology maps
    for (const auto& subshape_idx_pair : _shape_to_idx) {
        TopoDS_Shape subshape = subshape_idx_pair.first;
        int idx = subshape_idx_pair.second;
        int parent = topology.pk_to_idx[idx];
        int child;

        switch (subshape.ShapeType()) {
        case TopAbs_FACE:
            TopExp::MapShapes(subshape, TopAbs_WIRE, subshape_map);
            for (auto iterator = subshape_map.cbegin(); iterator != subshape_map.cend(); iterator++) {
                TopoDS_Shape subsubshape = *iterator;
                child = topology.pk_to_idx[_shape_to_idx[subsubshape]];

                loop_to_face[child] = parent;
                topology.face_to_loop.emplace_back(parent, child, PK_TOPOL_sense_none_c);
                face_to_loops[parent].push_back(child);
            }
            break;
        case TopAbs_WIRE:
            TopExp::MapShapes(subshape, TopAbs_EDGE, subshape_map);
            for (auto iterator = subshape_map.cbegin(); iterator != subshape_map.cend(); iterator++) {
                TopoDS_Shape subsubshape = *iterator;
                child = topology.pk_to_idx[_shape_to_idx[subsubshape]];

                // TODO: confirm this setting of sense matches relationship
                PK_TOPOL_sense_t sense =
                    subsubshape.Orientation() == TopAbs_FORWARD ? PK_TOPOL_sense_positive_c :
                    subsubshape.Orientation() == TopAbs_REVERSED ? PK_TOPOL_sense_negative_c :
                    PK_TOPOL_sense_none_c;

                edge_to_loop[child].push_back(parent);
                topology.loop_to_edge.emplace_back(parent, child, sense);
                loop_to_edges[parent].push_back(child);
            }
            break;
        case TopAbs_EDGE:
            TopExp::MapShapes(subshape, TopAbs_VERTEX, subshape_map);
            for (auto iterator = subshape_map.cbegin(); iterator != subshape_map.cend(); iterator++) {
                TopoDS_Shape subsubshape = *iterator;
                child = topology.pk_to_idx[_shape_to_idx[subsubshape]];

                topology.edge_to_vertex.emplace_back(parent, child, PK_TOPOL_sense_none_c);
                edge_to_vertices[parent].push_back(child);
            }
            break;
        case TopAbs_VERTEX:
            break;
        default:
            break;
        }

        subshape_map.Clear();
    }

    // Also find Face-Face Edges
    for (auto edgeloop : edge_to_loop) {
        int edge = edgeloop.first;
        auto loops = edgeloop.second;
        assert(loops.size() == 2);
        int face1 = loop_to_face[loops[0]];
        int face2 = loop_to_face[loops[1]];
        topology.face_to_face.emplace_back(face1, face2, edge);
    }

    // Construct Adjacency List Maps

    // Compute Lower Face Adjacencies
    std::map<int, std::vector<int> > face_to_edges;
    std::map<int, std::vector<int> > face_to_vertices;
    for (int face = 0; face < topology.faces.size(); ++face) {
        std::set<int> edge_neighbors;
        std::set<int> vertex_neighbors;
        for (int loop : face_to_loops[face]) {
            for (int edge : loop_to_edges[loop]) {
                edge_neighbors.insert(edge);
                for (int vertex : edge_to_vertices[edge]) {
                    vertex_neighbors.insert(vertex);
                }
            }
        }
        for (int edge : edge_neighbors) {
            face_to_edges[face].push_back(edge);
        }
        for (int vertex : vertex_neighbors) {
            face_to_vertices[face].push_back(vertex);
        }
    }

    // Compute Lower Loop Adjacencies
    std::map<int, std::vector<int> > loop_to_vertices;
    for (int loop = 0; loop < topology.loops.size(); ++loop) {
        std::set<int> vertex_neighbors;
        for (int edge : loop_to_edges[loop]) {
            for (int vertex : edge_to_vertices[edge]) {
                vertex_neighbors.insert(vertex);
            }
        }
        for (int vertex : vertex_neighbors) {
            loop_to_vertices[loop].push_back(vertex);
        }
    }

    // Assign to structure
    // TODO - don't use the temporary variables
    // TODO - int -> size_t
    topology.face_loop = face_to_loops;
    topology.face_edge = face_to_edges;
    topology.face_vertex = face_to_vertices;
    topology.loop_edge = loop_to_edges;
    topology.loop_vertex = loop_to_vertices;
    topology.edge_vertex = edge_to_vertices;

    return topology;
}

MassProperties OCCTBody::GetMassProperties(double accuracy) {
    // TODO: implement
    return MassProperties(0, accuracy);
}

Eigen::MatrixXd OCCTBody::GetBoundingBox() {
    Bnd_Box box;
    BRepBndLib::Add(_shape, box);

    double xmin, ymin, zmin, xmax, ymax, zmax;
    box.Get(xmin, ymin, zmin, xmax, ymax, zmax);

    Eigen::MatrixXd corners(2, 3);
    corners <<
        xmin, ymin, zmin,
        xmax, ymax, zmax;
    
    return corners;
}

int OCCTBody::Transform(const Eigen::MatrixXd& xfrm) {
    // TODO: implement
    return 0;
}

void OCCTBody::Tesselate(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::VectorXi& FtoT,
    Eigen::MatrixXi& EtoT,
    Eigen::VectorXi& VtoT) {
    // Setup faceting call options
    // TODO: give control over linear deflection
    double linear_deflection = 0.1;

    // Facet the body
    BRepMesh_IncrementalMesh(_shape, linear_deflection);

    // Gather vertices and triangles from faces of _shape
    std::unordered_map<gp_Pnt, int, gp_Pnt_Hash, gp_Pnt_Pred> pnt_idxs;
    std::vector<gp_Pnt> pnts;
    std::vector<Eigen::Vector3i> tris;

    TopLoc_Location loc;

    for (const auto& subshape_idx_pair : _shape_to_idx) {
        TopoDS_Shape subshape = subshape_idx_pair.first;

        if (subshape.ShapeType() == TopAbs_FACE) {
            TopoDS_Face subface = TopoDS::Face(subshape);
            opencascade::handle<Poly_Triangulation> subface_triangulation =
                BRep_Tool::Triangulation(subface, loc);
            
            if (!subface_triangulation.IsNull()) {
                // Add new points
                for (int i = 1; i <= subface_triangulation->NbNodes(); ++i) {
                    gp_Pnt pnt = subface_triangulation->Node(i);
                    if (pnt_idxs.find(pnt) == pnt_idxs.end()) {
                        pnt_idxs[pnt] = pnts.size();
                        pnts.push_back(pnt);
                    }
                }

                // Add new triangles
                for (int i = 1; i <= subface_triangulation->NbTriangles(); ++i) {
                    Poly_Triangle tri = subface_triangulation->Triangle(i);
                    gp_Pnt pnt1 = subface_triangulation->Node(tri.Value(1));
                    gp_Pnt pnt2 = subface_triangulation->Node(tri.Value(2));
                    gp_Pnt pnt3 = subface_triangulation->Node(tri.Value(3));
                    int pnt1_idx = pnt_idxs[pnt1];
                    int pnt2_idx = pnt_idxs[pnt2];
                    int pnt3_idx = pnt_idxs[pnt3];
                    tris.emplace_back(pnt1_idx, pnt2_idx, pnt3_idx);
                }
            }
        }
    }

    // Populate Mesh Vertices
    V.resize(pnts.size(), 3);
    for (int i = 0; i < pnts.size(); ++i) {
        auto pnt = pnts[i];
        V(i, 0) = pnt.X();
        V(i, 1) = pnt.Y();
        V(i, 2) = pnt.Z();
    }

    // Populate Mesh Faces
    F.resize(tris.size(), 3);
    for (int i = 0; i < tris.size(); ++i) {
        auto vec = tris[i];
        F(i, 0) = vec(0);
        F(i, 1) = vec(1);
        F(i, 2) = vec(2);
    }

    // TODO: Populate Mesh to Topology References
}

void OCCTBody::debug() {
    // TODO: implement
}