//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	tiuque                                                                  //
//	A 3D mesh manipulator software.	        				//
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Copyright � 2011 Cristobal A. Navarro.					//
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

#include <cleap_private.h> // Include first to avoid GL/GLEW order compile errors
#include "Application.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>
#include <iostream>

#include "validation.hpp" // checkCuda
using namespace mimir;

Application::Application(){
    myMesh = 0;

    ViewerOptions viewer_opts;
    viewer_opts.window.title  = "Tiuque"; // Top-level window.
    viewer_opts.window.size   = {1920, 1080};
    viewer_opts.present.mode  = PresentMode::VSync;
    viewer_opts.report_period = 180;
    viewer_opts.bg_color      = {0.f,0.f,0.f,1.f};
    engine.init(viewer_opts);

    // TODO: Fix dptr in kernels
	this->myMesh = new Mesh("/home/francisco/code/files_cudaview/bunny.off");
    auto m = this->myMesh->my_cleap_mesh;

    // NOTE: Cudaview code
    // TODO: Delete views
    // TODO: Add stride to attribute params
    auto vertices  = engine.allocLinear((void**)&m->dm->d_vbo_v, sizeof(float4) * cleap_get_vertex_count(m));
    auto triangles = engine.allocLinear((void**)&m->dm->d_eab, sizeof(int3) * cleap_get_face_count(m));

    ViewParams2 params;
    params.element_count = cleap_get_vertex_count(m);
    params.data_domain   = DataDomain::Domain3D;
    params.view_type     = ViewType::Markers;
    params.attributes[AttributeType::Position] = {
        .allocation = vertices,
        .format     = { .type = DataType::float32, .components = 3 }
    };
    engine.createView(params);

    // Recycle the above parameters, changing only what is needed
    params.element_count = cleap_get_face_count(m);
    params.view_type     = ViewType::Edges;
    params.indexing = {
        .allocation = triangles,
        .format     = { .type = DataType::int32, .components = 3 }
    };
    engine.createView(params);

    /*
    MemoryParams mem1;
    mem1.layout          = DataLayout::Layout1D;
    mem1.element_count.x = cleap_get_vertex_count(m);
    mem1.component_type  = ComponentType::Float;
    mem1.channel_count   = 4;
    mem1.resource_type   = ResourceType::Buffer;
    auto vertices = engine.createBuffer((void**)&m->dm->d_vbo_v, mem1);

    ViewParams params;
    params.element_count = cleap_get_vertex_count(m);
    params.data_domain   = DataDomain::Domain3D;
    params.domain_type   = DomainType::Unstructured;
    params.view_type     = ViewType::Markers;
    params.attributes[AttributeType::Position] = *vertices;
    engine.createView(params);

    MemoryParams mem2;
    mem2.layout          = DataLayout::Layout1D;
    mem2.element_count.x = cleap_get_face_count(m);
    mem2.component_type  = ComponentType::Int;
    mem2.channel_count   = 3;
    mem2.resource_type   = ResourceType::IndexBuffer;
    auto triangles = engine.createBuffer((void**)&m->dm->d_eab, mem2);

    params.element_count = cleap_get_face_count(m);
    params.view_type     = ViewType::Edges;
    params.attributes[AttributeType::Index] = *triangles;
    params.options.default_color  = {0.f, 1.f, 0.f, 1.f};
    engine.createView(params);*/

    checkCuda(cudaMemcpy(m->dm->d_vbo_v, m->vnc_data.v,
        cleap_get_vertex_count(m) * sizeof(float3), cudaMemcpyHostToDevice)
    );
    checkCuda(cudaMemcpy(m->dm->d_eab, m->triangles,
        cleap_get_face_count(m) * sizeof(uint3), cudaMemcpyHostToDevice)
    );
}

Application::~Application(){
}

void Application::on_button_exit_clicked(){
    //Gtk::Main::quit();
}

void Application::on_menu_help_about(){
    //about_dialog->run();
    //about_dialog->hide();
}

void Application::on_button_delaunay_2d_clicked(){
    if(myMesh){
        //engine.prepareViews();
        if (educational_mode) //if(check_button_educational_mode->get_active())
        	myMesh->delaunay_transformation_interactive(CLEAP_MODE_2D);
        else
            myMesh->delaunay_transformation(CLEAP_MODE_2D);

        //engine.updateViews(); ////my_gl_window->redraw();
    }
}

void Application::on_button_clear_clicked(){
    delete myMesh;
    myMesh = 0;
    //my_gl_window->set_mesh_pointer(myMesh);
    //my_gl_window->redraw();
}

void Application::on_button_delaunay_3d_clicked(){
    if(myMesh){
        //engine.prepareViews();
        if (educational_mode) //if(check_button_educational_mode->get_active())
            myMesh->delaunay_transformation_interactive(CLEAP_MODE_3D);
        else
            myMesh->delaunay_transformation(CLEAP_MODE_3D);

        //engine.updateViews(); //my_gl_window->redraw();
    }
}

void Application::on_toggle_button_wireframe_toggled(){
    if(myMesh){
        myMesh->set_solid(static_cast<int>(toggle_wireframe));
        //myMesh->set_wireframe((int)toogle_button_wireframe->get_active());
        //my_gl_window->redraw();
    }
}

void Application::on_toggle_button_solid_toggled(){
    if(myMesh){
        myMesh->set_solid(static_cast<int>(toggle_solid));
        //myMesh->set_solid((int)toogle_button_solid->get_active());
        //my_gl_window->redraw();
    }
}

void Application::on_check_button_educational_mode_toggled(){

}

void Application::on_hscale_size_change_value(){
    //my_gl_window->set_render_scale(hscale_size->get_value());
    //my_gl_window->redraw();
}

void Application::init()
{
    /*if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open", "Ctrl+O"))
            {
                ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".off", ".");
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            std::string file_name = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string curr_path = ImGuiFileDialog::Instance()->GetCurrentPath();
            //file_handler(file_name, curr_path);
        }
        ImGuiFileDialog::Instance()->Close();
    }*/

    engine.setGuiCallback([=,this]
    {
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open Mesh", "Ctrl+O"))
                {
                    ImGuiFileDialog::Instance()->OpenDialog("LoadFileDialog", "Load mesh", ".off");
                }
                if (ImGui::MenuItem("Save", "Ctrl+S"))
                {
                    on_menu_file_save();
                }
                if (ImGui::MenuItem("Save As.."))
                {
                    ImGuiFileDialog::Instance()->OpenDialog("SaveFileDialog", "Save mesh", ".off");
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Quit", "Alt+F4"))
                {
                    engine.exit();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Run"))
            {
                if (ImGui::MenuItem("MDT-2D", "Ctrl+2"))
                {
                    on_button_delaunay_2d_clicked();
                }
                if (ImGui::MenuItem("MDT-3D", "Ctrl+3"))
                {
                    on_button_delaunay_3d_clicked();
                }
                ImGui::Separator();
                if (ImGui::BeginMenu("Options"))
                {
                    // TODO: Handle flag value changes
                    ImGui::MenuItem("Toggle Wireframe", "", &toggle_wireframe);
                    ImGui::MenuItem("Educational Mode", "", &educational_mode);
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (ImGuiFileDialog::Instance()->Display("LoadFileDialog"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();
                load_mesh(filename.c_str());
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("SaveFileDialog"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                std::string filename = ImGuiFileDialog::Instance()->GetFilePathName();
                save_mesh(filename.c_str());
            }
            ImGuiFileDialog::Instance()->Close();
        }
    });
    engine.displayAsync();

    //! linking widgets to logic
    // gl window -- important to be first
    /*ref_builder->get_widget_derived("widget_gl_window", //my_gl_window );
    if(//my_gl_window){
        printf("GL Initialized\n");
    }
    // quit button
    ref_builder->get_widget("widget_button_exit", button_exit);
    if(button_exit){
        button_exit->signal_clicked().connect( sigc::mem_fun(*this, &Application::on_button_exit_clicked) );
    }
    // clear button
    ref_builder->get_widget("widget_button_clear", button_clear);
    if(button_clear){
        button_clear->signal_clicked().connect( sigc::mem_fun(*this, &Application::on_button_clear_clicked) );
    }
    // mdt 2d
    ref_builder->get_widget("widget_button_mdt_2d", button_mdt_2d);
    if(button_mdt_2d){
        button_mdt_2d->signal_clicked().connect(sigc::mem_fun(*this, &Application::on_button_delaunay_2d_clicked));
    }
    // mdt 3d
    ref_builder->get_widget("widget_button_mdt_3d", button_mdt_3d);
    if(button_mdt_3d){
        button_mdt_3d->signal_clicked().connect(sigc::mem_fun(*this, &Application::on_button_delaunay_3d_clicked));
    }
    // toogle wireframe
    ref_builder->get_widget("widget_toggle_button_wireframe", toogle_button_wireframe);
    if( toogle_button_wireframe){
        toogle_button_wireframe->signal_clicked().connect(sigc::mem_fun(*this, &Application::on_toggle_button_wireframe_toggled));
    }
    // toogle solid
    ref_builder->get_widget("widget_toggle_button_solid", toogle_button_solid);
    if( toogle_button_solid){
        toogle_button_solid->signal_clicked().connect(sigc::mem_fun(*this, &Application::on_toggle_button_solid_toggled));
    }

    // hscale size
    ref_builder->get_widget("widget_hscale_size", hscale_size);
    if( hscale_size){
        hscale_size->signal_value_changed().connect(sigc::mem_fun(*this, &Application::on_hscale_size_change_value));
        hscale_size->set_range(0.1, 10);
        hscale_size->set_value(1.0);
    }
    // menu -> file -> open
    ref_builder->get_widget("widget_menubar_file_open", item_open);
    if(item_open){
        item_open->signal_activate().connect( sigc::mem_fun(*this, &Application::on_menu_file_open) );
    }
    // menu -> file -> save_as
    ref_builder->get_widget("widget_menubar_file_save_as", item_save_as);
    if(item_save_as){
        item_save_as->signal_activate().connect( sigc::mem_fun(*this, &Application::on_menu_file_save_as) );
    }
    // menu -> file -> save
    ref_builder->get_widget("widget_menubar_file_save", item_save);
    if(item_save){
        item_save->signal_activate().connect( sigc::mem_fun(*this, &Application::on_menu_file_save) );
    }
    // menu -> about dialog
    ref_builder->get_widget("widget_menubar_help_about", item_help_about);
    ref_builder->get_widget("widget_about_dialog", about_dialog);
    if(item_help_about && about_dialog){
        item_help_about->signal_activate().connect( sigc::mem_fun(*this, &Application::on_menu_help_about) );
    }
    ref_builder->get_widget("widget_check_button_educational_mode", check_button_educational_mode);
    if(check_button_educational_mode){
        check_button_educational_mode->signal_activate().connect(sigc::mem_fun(*this, &Application::on_check_button_educational_mode_toggled));
    }*/
}

void Application::on_menu_file_open()
{
/*
	Gtk::FileChooserDialog *dialog = new Gtk::FileChooserDialog("Open Mesh", Gtk::FILE_CHOOSER_ACTION_OPEN);
  	dialog->set_transient_for(*this);
  	//Add response buttons the the dialog:
  	dialog->add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
  	dialog->add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);
  	//Add filters, so that only certain file types can be selected:
  	Gtk::FileFilter *filter_geomview = new Gtk::FileFilter();
  	filter_geomview->set_name("off (Geomview)");
  	filter_geomview->add_pattern("*.off");
  	dialog->add_filter(*filter_geomview);

  	//Show the dialog and wait for a user response:
  	int result = dialog->run();
	if (result==Gtk::RESPONSE_OK){
		dialog->hide();
		Gtk::FileFilter *filtro=dialog->get_filter();
		if (filtro->get_name()=="off (Geomview)"){
			string file_string = dialog->get_filename();
			this->myMesh = new Mesh(file_string.c_str());
			on_toggle_button_solid_toggled();
			on_toggle_button_wireframe_toggled();
			//this->my_gl_window->set_mesh_pointer(myMesh);
			//this->my_gl_window->set_camera_from_mesh_pointer();
			//my_gl_window->redraw();
		}
    }

	delete filter_geomview;
	delete dialog;
	//my_gl_window->redraw();
*/
}

void Application::on_menu_file_save_as()
{
    printf("Tiuque::save_as::");
    if(this->myMesh){
        /*Gtk::FileChooserDialog dialog("Save as", Gtk::FILE_CHOOSER_ACTION_SAVE);
        dialog.set_transient_for(*this);
        //Add response buttons the the dialog:
        dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
        dialog.add_button(Gtk::Stock::SAVE, Gtk::RESPONSE_OK);
        //Add filters, so that only certain file types can be selected:
        Gtk::FileFilter filter_geomview;
        filter_geomview.set_name("off (Geomview)");
        filter_geomview.add_pattern("*.off");
        dialog.add_filter(filter_geomview);
        int result=dialog.run();
        if (result==Gtk::RESPONSE_OK){
            myMesh->save_mesh(dialog.get_filename().c_str());
            printf("ok\n");
        }
        else{
            printf("cancel\n");
        }*/
        return;
    }
    else{
        printf("nothing to save... have you loaded a mesh?.\n");
        return;
    }
}


void Application::on_menu_file_save(){

    printf("Tiuque::save::");
    if(this->myMesh){
        myMesh->save_mesh_default();
        printf("ok\n");
    }
    else{
        printf("nothing to save... have you loaded a mesh?.\n");
        return;
    }
}

int Application::load_mesh(const char* filename)
{
    /*this->myMesh = new Mesh(filename);
    auto m = this->myMesh->my_cleap_mesh;

    validation::checkCuda(cudaMemcpy(d_vertices, m->vnc_data.v,
        cleap_get_vertex_count(m) * sizeof(float3), cudaMemcpyHostToDevice)
    );
    validation::checkCuda(cudaMemcpy(d_triangles, m->triangles,
        cleap_get_face_count(m) * sizeof(uint3), cudaMemcpyHostToDevice)
    );*/
    return 0;
}

int Application::save_mesh(const char* filename)
{
    printf("Tiuque::save_as::");
    myMesh->save_mesh(filename);
    printf("ok\n");
    return 0;
}
