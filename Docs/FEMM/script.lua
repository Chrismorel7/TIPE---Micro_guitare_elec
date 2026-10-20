local frequency = 78
local sample_count = 100
local duration = 0.04
local pi = 3.1415926535
local previous_displacement = 0
local mode_amplitudes = {
    1.09, 0.47, 0.24, 0.11, 0.045, 0.001, -0.021, -0.029, -0.027, -0.019
}
local output_path = "resultats_flux.txt"
local problem_name = "Humbucker.FEM"

if sample_count < 1 or duration <= 0 or frequency <= 0 then
    error("Sample count, duration, and frequency must be positive")
end

mi_saveas(problem_name)
mi_setfocus(problem_name)

local output_file = openfile(output_path, "w")
if output_file == nil then
    error("Unable to open FEMM output file: " .. output_path)
end

write(output_file, "Time_s, Displacement_mm, Flux_Weber\n")

for sample_index = 0, sample_count do
    local time = duration * sample_index / sample_count
    local displacement = 0

    for mode_index = 1, table.getn(mode_amplitudes) do
        displacement = displacement
            + mode_amplitudes[mode_index]
                * sin(2 * pi * mode_index * frequency * time)
    end

    local displacement_step = displacement - previous_displacement
    mi_selectgroup(1)
    mi_movetranslate(0, displacement_step)
    mi_clearselected()
    local analysis_status = mi_analyze(1)
    if analysis_status == 0 then
        closefile(output_file)
        error("FEMM failed to solve sample " .. sample_index)
    end
    mi_loadsolution()

    local _, _, flux = mo_getcircuitproperties("Bobine")
    if flux == nil then
        mo_close()
        closefile(output_file)
        error("Unable to read coil flux from the FEMM solution")
    end

    write(
        output_file,
        format("%g, %g, %g\n", time, displacement, flux)
    )
    mo_close()
    previous_displacement = displacement
end

closefile(output_file)
