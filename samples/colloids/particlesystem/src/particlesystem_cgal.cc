#define CGAL_DISABLE_ROUNDING_MATH_CHECK

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_2.h>
#include <CGAL/Periodic_2_Delaunay_triangulation_traits_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Timer.h>

#include <cassert>
#include <unordered_map>
#include <vector>

#include "particlesystem_delaunay.h"

namespace particlesystem {
namespace delaunay {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Periodic_2_Delaunay_triangulation_traits_2<K> GT;
typedef CGAL::Periodic_2_triangulation_vertex_base_2<GT>    Vb;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, GT, Vb> VbInfo;
typedef CGAL::Periodic_2_triangulation_face_base_2<GT>      Fb;
typedef CGAL::Triangulation_data_structure_2<VbInfo, Fb>    Tds;
typedef CGAL::Periodic_2_Delaunay_triangulation_2<GT, Tds>  PDT;

typedef PDT::Face_iterator Face_iterator;
typedef PDT::Face_handle   Face_handle;
typedef PDT::Triangle      Triangle;
typedef PDT::Point         Point;
typedef PDT::Iso_rectangle Iso_rectangle;

/* Auxiliary edge structure, used during construction of the edge list.
 */
struct temp_edge
{
	int id;
	int n1, n2;
	int a1, a2;
	int b1, b2;
	int op1, op2;
};

void ParticleSystemDelaunay::initTriangulationHost()
{
	CGAL::Timer t;

	// The domain is a 2D box with origins at the bottom left corner.
	double boxLength = params_.boxlength;
	Iso_rectangle hDomain(0, 0, boxLength, boxLength);

	// The point list contains the coordinates of each point and their
	// respective index on the original array.
	std::vector<std::pair<Point, unsigned int>> hPointList;
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		Point p = Point(positions_[2*i], positions_[2*i+1]);
		hPointList.push_back(std::make_pair(p, i));
	}

	std::cout << "Building triangulation in host memory..." << std::endl;
	// Creation of the triangulation:
	PDT triangulation(hDomain);
	t.start();
	// Iterator range insertion using spatial sorting and dummy point heuristic
	triangulation.insert(hPointList.begin(), hPointList.end(), true);
	t.stop();
	std::cout << "[INFO] CGAL::triangulation.insert() = " << t.time()
			  << " seconds" << std::endl;

	// Check if the triangulation can be used by the CUDA functions
	assert(triangulation.is_triangulation_in_1_sheet());
	assert(params_.num_elements == triangulation.number_of_vertices());

	delaunay_.num_triangles = triangulation.number_of_faces();
	delaunay_.num_edges = triangulation.number_of_edges();

	// Host triangles array
	delaunay_.triangles = new unsigned int[delaunay_.num_triangles * 3];

	// the hash for edges
	std::unordered_map<int, std::unordered_map<int, temp_edge> > root_hash;
	std::unordered_map<int, temp_edge>::iterator hit;

	std::vector<temp_edge*> edge_vector;

	temp_edge* aux_tmp_edge;
	int j_sec[3] = {0, 0, 1};
	int k_sec[3] = {1, 2, 2};
	int op_sec[3] = {2, 1, 0};
	int j, k, op;

	// Iteration through the faces of the triangulation, adding edges to the
	// list if they have not been visited yet.
	uint triangleIndex = 0;
	for (Face_iterator faceIter = triangulation.faces_begin();
		 faceIter != triangulation.faces_end();
		 faceIter++)
	{
		Face_handle pFace = faceIter;
		for (int i = 0; i < 3; i++)
		{
			delaunay_.triangles[triangleIndex + i] = pFace->vertex(i)->info();
		}

		for (int q = 0; q < 3; q++)
		{
			j = j_sec[q], k = k_sec[q], op = op_sec[q];
			// always the higher first
			if (delaunay_.triangles[triangleIndex + j] < delaunay_.triangles[triangleIndex + k])
			{
				k = j;
				j = k_sec[q];
			}
			// ok, first index already existed, check if the second exists or not
			std::unordered_map<int, temp_edge> *second_hash = &root_hash[delaunay_.triangles[triangleIndex + j]];
			hit = second_hash->find(delaunay_.triangles[triangleIndex + k]);
			if (hit != second_hash->end())
			{
				// the edge already exists, then fill the remaining info
				aux_tmp_edge = &(hit->second);
				aux_tmp_edge->b1 = triangleIndex + j;
				aux_tmp_edge->b2 = triangleIndex + k;
				aux_tmp_edge->op2 = triangleIndex + op;
			}
			else
			{
				// create a new edge
				aux_tmp_edge = &(*second_hash)[delaunay_.triangles[triangleIndex + k]]; // create the low value on secondary_hash
				aux_tmp_edge->n1 = delaunay_.triangles[triangleIndex + j];
				aux_tmp_edge->n2 = delaunay_.triangles[triangleIndex + k];
				aux_tmp_edge->a1 = triangleIndex + j;
				aux_tmp_edge->a2 = triangleIndex + k;
				aux_tmp_edge->b1 = -1;
				aux_tmp_edge->b2 = -1;
				aux_tmp_edge->op1 = triangleIndex + op;
				aux_tmp_edge->op2 = -1;
				aux_tmp_edge->id = edge_vector.size();
				edge_vector.push_back( aux_tmp_edge );
			}
		}

		triangleIndex = triangleIndex + 3;
	}
	assert(triangleIndex == delaunay_.num_triangles * 3);
	assert(edge_vector.size() == delaunay_.num_edges);

	// Creation of separate edge arrays. The edge list is unzipped to accomodate
	// the data structure on device memory.
	delaunay_.edge_idx = new int2[delaunay_.num_edges];
	delaunay_.edge_ta = new int2[delaunay_.num_edges];
	delaunay_.edge_tb = new int2[delaunay_.num_edges];
	delaunay_.edge_op = new int2[delaunay_.num_edges];

	for (unsigned int i = 0; i < delaunay_.num_edges; i++)
	{
		delaunay_.edge_idx[i] = make_int2(edge_vector[i][0].n1, edge_vector[i][0].n2);
		delaunay_.edge_ta[i] = make_int2(edge_vector[i][0].a1, edge_vector[i][0].a2);
		delaunay_.edge_tb[i] = make_int2(edge_vector[i][0].b1, edge_vector[i][0].b2);
		delaunay_.edge_op[i] = make_int2(edge_vector[i][0].op1, edge_vector[i][0].op2);
	}

	std::cout << "[INFO] Loaded triangulation on host memory." << std::endl;
}

} // namespace delaunay
} // namespace particlesystem
