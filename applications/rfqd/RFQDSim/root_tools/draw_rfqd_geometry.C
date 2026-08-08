#include <TCanvas.h>
#include <TColor.h>
#include <TEllipse.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPad.h>
#include <TPolyLine3D.h>
#include <TStyle.h>
#include <TString.h>
#include <TTree.h>

#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

namespace {

void draw3DLine(const std::vector<std::array<double, 3>>& points, Color_t color,
                Width_t width = 1, Style_t style = 1) {
    auto* line = new TPolyLine3D(static_cast<int>(points.size()));
    for (std::size_t index = 0; index < points.size(); ++index) {
        line->SetPoint(static_cast<int>(index), points[index][0], points[index][1], points[index][2]);
    }
    line->SetLineColor(color);
    line->SetLineWidth(width);
    line->SetLineStyle(style);
    line->Draw("same");
}

void drawHyperbolicElectrodes(double r0_mm, double z_min_mm, double z_max_mm) {
    constexpr int n_transverse = 33;
    constexpr int n_longitudinal = 11;
    const double transverse_limit = 1.25 * r0_mm;

    for (int family = 0; family < 2; ++family) {
        const Color_t color = family == 0 ? kRed + 1 : kBlue + 1;
        for (int sign : {-1, 1}) {
            for (int transverse_index = 0; transverse_index < n_transverse; ++transverse_index) {
                const double u = -transverse_limit
                        + 2.0 * transverse_limit * transverse_index / (n_transverse - 1);
                const double hyperbolic_coordinate = sign * std::sqrt(r0_mm * r0_mm + u * u);
                std::vector<std::array<double, 3>> longitudinal_line;
                if (family == 0) {
                    longitudinal_line = {
                        {z_min_mm, hyperbolic_coordinate, u},
                        {z_max_mm, hyperbolic_coordinate, u}
                    };
                }
                else {
                    longitudinal_line = {
                        {z_min_mm, u, hyperbolic_coordinate},
                        {z_max_mm, u, hyperbolic_coordinate}
                    };
                }
                draw3DLine(longitudinal_line, color);
            }

            for (int longitudinal_index = 0; longitudinal_index < n_longitudinal; ++longitudinal_index) {
                const double z = z_min_mm
                        + (z_max_mm - z_min_mm) * longitudinal_index / (n_longitudinal - 1);
                std::vector<std::array<double, 3>> transverse_line;
                transverse_line.reserve(n_transverse);
                for (int transverse_index = 0; transverse_index < n_transverse; ++transverse_index) {
                    const double u = -transverse_limit
                            + 2.0 * transverse_limit * transverse_index / (n_transverse - 1);
                    const double hyperbolic_coordinate = sign * std::sqrt(r0_mm * r0_mm + u * u);
                    if (family == 0) {
                        transverse_line.push_back({z, hyperbolic_coordinate, u});
                    }
                    else {
                        transverse_line.push_back({z, u, hyperbolic_coordinate});
                    }
                }
                draw3DLine(transverse_line, color);
            }
        }
    }
}

TGraph* makeHyperbolaGraph(double r0_mm, int family, int sign) {
    constexpr int n_points = 161;
    const double transverse_limit = 1.25 * r0_mm;
    auto* graph = new TGraph(n_points);
    for (int index = 0; index < n_points; ++index) {
        const double u = -transverse_limit
                + 2.0 * transverse_limit * index / (n_points - 1);
        const double hyperbolic_coordinate = sign * std::sqrt(r0_mm * r0_mm + u * u);
        if (family == 0) {
            graph->SetPoint(index, hyperbolic_coordinate, u);
            graph->SetLineColor(kRed + 1);
        }
        else {
            graph->SetPoint(index, u, hyperbolic_coordinate);
            graph->SetLineColor(kBlue + 1);
        }
    }
    graph->SetLineWidth(2);
    return graph;
}

} // namespace

void draw_rfqd_geometry(const char* root_file_name = "rfqd-ideal.root",
                        const char* output_name = "rfqd_geometry.png",
                        int maximum_tracks = 12) {
    gStyle->SetOptStat(0);

    TFile* input_file = TFile::Open(root_file_name, "READ");
    if (!input_file || input_file->IsZombie()) {
        std::cerr << "Cannot open ROOT file: " << root_file_name << std::endl;
        return;
    }

    auto* config_tree = dynamic_cast<TTree*>(input_file->Get("run_config"));
    auto* trajectory_tree = dynamic_cast<TTree*>(input_file->Get("trajectory"));
    auto* summary_tree = dynamic_cast<TTree*>(input_file->Get("particle_summary"));
    if (!config_tree || !trajectory_tree || !summary_tree) {
        std::cerr << "Required TTrees are missing (run_config, trajectory, particle_summary)" << std::endl;
        return;
    }

    double r0_m = 0.0;
    double z_min_m = 0.0;
    double z_max_m = 0.0;
    double v_rf = 0.0;
    double frequency_rf = 0.0;
    config_tree->SetBranchAddress("r0_m", &r0_m);
    config_tree->SetBranchAddress("rfqd_z_min_m", &z_min_m);
    config_tree->SetBranchAddress("rfqd_z_max_m", &z_max_m);
    config_tree->SetBranchAddress("V_rf", &v_rf);
    config_tree->SetBranchAddress("frequency_rf", &frequency_rf);
    config_tree->GetEntry(0);

    const double r0_mm = r0_m * 1.0e3;
    const double z_min_mm = z_min_m * 1.0e3;
    const double z_max_mm = z_max_m * 1.0e3;
    const double transverse_limit_mm = 2.05 * r0_mm;

    auto* canvas = new TCanvas("rfqd_geometry", "RFQD model geometry", 1600, 720);
    canvas->Divide(2, 1);

    canvas->cd(1);
    auto* frame3d = new TH3D(
            "rfqd_3d_frame", ";z [mm];x [mm];y [mm]",
            2, z_min_mm, z_max_mm,
            2, -transverse_limit_mm, transverse_limit_mm,
            2, -transverse_limit_mm, transverse_limit_mm);
    frame3d->Draw();
    drawHyperbolicElectrodes(r0_mm, z_min_mm, z_max_mm);

    Int_t global_index = 0;
    Double_t x_m = 0.0;
    Double_t y_m = 0.0;
    Double_t z_m = 0.0;
    trajectory_tree->SetBranchAddress("global_index", &global_index);
    trajectory_tree->SetBranchAddress("x_m", &x_m);
    trajectory_tree->SetBranchAddress("y_m", &y_m);
    trajectory_tree->SetBranchAddress("z_m", &z_m);

    std::map<int, std::vector<std::array<double, 3>>> trajectories;
    for (Long64_t entry = 0; entry < trajectory_tree->GetEntries(); ++entry) {
        trajectory_tree->GetEntry(entry);
        if (global_index >= 0 && global_index < maximum_tracks) {
            trajectories[global_index].push_back({z_m * 1.0e3, x_m * 1.0e3, y_m * 1.0e3});
        }
    }

    const Color_t trajectory_colors[] = {
        kBlack, kGreen + 2, kMagenta + 1, kOrange + 7, kCyan + 2,
        kViolet + 1, kGray + 2, kAzure + 1
    };
    int color_index = 0;
    for (const auto& [particle_index, points] : trajectories) {
        (void)particle_index;
        draw3DLine(points, trajectory_colors[color_index % 8], 2);
        ++color_index;
    }
    gPad->SetTheta(18.0);
    gPad->SetPhi(-55.0);
    gPad->Modified();

    auto* label3d = new TLatex();
    label3d->SetNDC();
    label3d->SetTextSize(0.028);
    label3d->DrawLatex(0.12, 0.92, Form("Ideal analytic RFQD: V_{rf}=%.3g V, f=%.3g MHz",
                                      v_rf, frequency_rf * 1.0e-6));
    label3d->DrawLatex(0.12, 0.875, "red/blue: opposite RF electrode families; black/colored: trajectories");

    canvas->cd(2);
    auto* frame2d = new TH2D(
            "rfqd_xy_frame", "Ideal RFQD cross section;x [mm];y [mm]",
            100, -transverse_limit_mm, transverse_limit_mm,
            100, -transverse_limit_mm, transverse_limit_mm);
    frame2d->Draw();

    auto* aperture = new TEllipse(0.0, 0.0, r0_mm, r0_mm);
    aperture->SetFillStyle(0);
    aperture->SetLineColor(kGray + 2);
    aperture->SetLineStyle(2);
    aperture->SetLineWidth(2);
    aperture->Draw("same");

    TGraph* x_electrode_legend = nullptr;
    TGraph* y_electrode_legend = nullptr;
    for (int family = 0; family < 2; ++family) {
        for (int sign : {-1, 1}) {
            auto* graph = makeHyperbolaGraph(r0_mm, family, sign);
            graph->Draw("L same");
            if (family == 0 && sign == 1) {
                x_electrode_legend = graph;
            }
            if (family == 1 && sign == 1) {
                y_electrode_legend = graph;
            }
        }
    }

    Int_t summary_global_index = 0;
    Int_t termination_code = 0;
    Double_t x_start_m = 0.0;
    Double_t y_start_m = 0.0;
    Double_t x_final_m = 0.0;
    Double_t y_final_m = 0.0;
    summary_tree->SetBranchAddress("global_index", &summary_global_index);
    summary_tree->SetBranchAddress("termination_code", &termination_code);
    summary_tree->SetBranchAddress("x_start_m", &x_start_m);
    summary_tree->SetBranchAddress("y_start_m", &y_start_m);
    summary_tree->SetBranchAddress("x_final_m", &x_final_m);
    summary_tree->SetBranchAddress("y_final_m", &y_final_m);

    auto* initial_positions = new TGraph();
    auto* transmitted_positions = new TGraph();
    for (Long64_t entry = 0; entry < summary_tree->GetEntries(); ++entry) {
        summary_tree->GetEntry(entry);
        initial_positions->SetPoint(initial_positions->GetN(), x_start_m * 1.0e3, y_start_m * 1.0e3);
        if (termination_code == 1) {
            transmitted_positions->SetPoint(
                    transmitted_positions->GetN(), x_final_m * 1.0e3, y_final_m * 1.0e3);
        }
    }
    initial_positions->SetMarkerStyle(24);
    initial_positions->SetMarkerSize(0.8);
    initial_positions->SetMarkerColor(kBlack);
    initial_positions->Draw("P same");
    transmitted_positions->SetMarkerStyle(20);
    transmitted_positions->SetMarkerSize(0.8);
    transmitted_positions->SetMarkerColor(kGreen + 2);
    transmitted_positions->Draw("P same");

    auto* legend = new TLegend(0.12, 0.70, 0.48, 0.89);
    legend->AddEntry(x_electrode_legend, "x-electrode equipotential", "l");
    legend->AddEntry(y_electrode_legend, "y-electrode equipotential", "l");
    legend->AddEntry(aperture, "r_{0} termination aperture", "l");
    legend->AddEntry(initial_positions, "initial particles", "p");
    legend->AddEntry(transmitted_positions, "transmitted particles", "p");
    legend->Draw();

    auto* caveat = new TLatex();
    caveat->SetNDC();
    caveat->SetTextSize(0.028);
    caveat->DrawLatex(0.12, 0.94, "Hyperbolic equipotential model; not a cylindrical-rod mechanical drawing");

    canvas->Modified();
    canvas->Update();
    canvas->SaveAs(output_name);
    std::cout << "Saved geometry view: " << output_name << std::endl;
}
