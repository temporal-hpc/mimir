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

#include "Mesh.h"
Mesh::Mesh(){
    init();
}
Mesh::Mesh(cleap_mesh* m){
    init();
    this->my_cleap_mesh = m;
}
// These should be handled by cudaview now
Mesh::Mesh(const char *filename){
    cleap_init();
    this->my_cleap_mesh = cleap_load_mesh(filename);
    this->default_filename = filename;
}
void Mesh::init(){
    this->my_cleap_mesh = 0;
    this->default_filename = 0;
    this->accumflips=0;
}
void Mesh::save_mesh(const char *filename){
    cleap_save_mesh(my_cleap_mesh, filename);
}
void Mesh::save_mesh_default(){
    if(this->default_filename)
        cleap_save_mesh(my_cleap_mesh, this->default_filename);
    else
        printf("Tiuque::save_mesh_default::cannot save... default_filename = NULL.");
}
void Mesh::extern_render_mesh(){
    cleap_render_mesh(my_cleap_mesh);
}
void Mesh::delaunay_transformation(int mode){
    cleap_delaunay_transformation(this->my_cleap_mesh, mode);
}
void Mesh::delaunay_transformation_interactive(int mode){
	int flips = cleap_delaunay_transformation_interactive(this->my_cleap_mesh, mode);
	accumflips += flips;
	printf("flips = %i       accumflips = %i\n", flips, accumflips);
}
void Mesh::print_mesh(){
    cleap_print_mesh(my_cleap_mesh);
}
Mesh::~Mesh(){
    cleap_clear_mesh(my_cleap_mesh);
    this->my_cleap_mesh = 0;
}