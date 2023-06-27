//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	tiuque                                                                  //
//	A 3D mesh manipulator software.	        				//
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Copyright © 2011 Cristobal A. Navarro.					//
//										//	
//	This file is part of tiuque.						//
//	tiuque is free software: you can redistribute it and/or modify		//
//	it under the terms of the GNU General Public License as published by	//
//	the Free Software Foundation, either version 3 of the License, or	//
//	(at your option) any later version.					//
//										//
//	tiuque is distributed in the hope that it will be useful,		//
//	but WITHOUT ANY WARRANTY; without even the implied warranty of		//
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	    	//
//	GNU General Public License for more details.				//
//										//
//	You should have received a copy of the GNU General Public License	//
//	along with tiuque.  If not, see <http://www.gnu.org/licenses/>. 	//
//										//
//////////////////////////////////////////////////////////////////////////////////

#ifndef MESH_H
#define MESH_H

#include <string>
#include <set>
#include <map>
#include <vector>
#include <stack>
#include <limits>
#include <cstdlib>
#include <cmath>

//#include "globals.h" // global
#include <cleap.h> // cleap library

class Mesh{
    private:
        //private fields
        cleap_mesh* my_cleap_mesh;
        const char* default_filename;
        int accumflips;
    public:
        //constructors & destructor
        Mesh();
        Mesh(const char* filename);
        Mesh(cleap_mesh* m);
        ~Mesh();
        //init function
        void init();
        void save_mesh(const char* filename);
        void save_mesh_default();
        void setCuantities( int numVertexes, int numFaces, int numEdges );

        void set_default_filename(char* filename){ this->default_filename = filename;}
        const char* get_default_filename(){ return this->default_filename; }

        int get_vertex_count(){ return cleap_get_vertex_count(this->my_cleap_mesh);}
        int get_face_count(){ return cleap_get_face_count(this->my_cleap_mesh);}
        int get_edge_count(){return cleap_get_edge_count(this->my_cleap_mesh);}

        float get_bsphere_x(){return cleap_get_bsphere_x(this->my_cleap_mesh); }
        float get_bsphere_y(){return cleap_get_bsphere_y(this->my_cleap_mesh); }
        float get_bsphere_z(){return cleap_get_bsphere_z(this->my_cleap_mesh); }
        float get_bsphere_r(){return cleap_get_bsphere_r(this->my_cleap_mesh); }

        bool is_wireframe(){ return cleap_mesh_is_wireframe(this->my_cleap_mesh); }
        bool is_solid(){ return cleap_mesh_is_solid(this->my_cleap_mesh); }
        void set_wireframe(int w){ cleap_mesh_set_wireframe(this->my_cleap_mesh, w); }
        void set_solid(int s){ cleap_mesh_set_solid(this->my_cleap_mesh, s);}

        void extern_render_mesh();
        void delaunay_transformation(int mode);
        void delaunay_transformation_interactive(int mode);

        void print_mesh();
};
#endif