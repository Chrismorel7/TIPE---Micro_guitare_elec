-- === SCRIPT ULTIME DE DEBUG ===
showconsole()
clearconsole()
print("Demarrage...")

-- 1. On definit explicitement le nom du fichier de travail
-- Cela permet au script de retrouver la fenêtre si elle est perdue
mi_saveas("Humucker.FEM")
nom_fichier_doc = "Humbucker.FEM"

-- 2. Paramètres
freq = 78
nb_points = 100
Te = 0.04
PI = 3.1415926535
pos_precedente = 0
Amplitudes = {1.09, 0.47, 0.24, 0.11, 0.045, 0.001, -0.021, -0.029, -0.027, -0.019}

-- 3. Fichier sortie
fich_sortie = "resultats_flux.txt"
handle = openfile(fich_sortie, "w")
write(handle, "Temps_s, Deplacement_mm, Flux_Weber\n")

-- 4. BOUCLE
for i = 0, nb_points do
    
    -- A. FORCER LE FOCUS SUR LE DESSIN (La solution miracle)
    -- On dit à FEMM : "Regarde le fichier de dessin, pas ailleurs"
    mi_setfocus(nom_fichier_doc)

    -- Remise à zero de la somme pour ce pas de temps
    deplacement_absolu = 0
    t = Te / i
    -- Boucle de sommation (Sigma)
    -- 'table.getn(Amplitudes)' renvoie la taille de la liste (ici 3)
    for n = 1, 10 do
        deplacement_absolu = deplacement_absolu + Amplitudes[n] * sin(2 * PI * n * freq * t)
    end

    mouvement_step = deplacement_absolu - pos_precedente
    
    -- C. Selection et Mouvement
    mi_selectgroup(1)
    mi_movetranslate(0, mouvement_step)
    mi_clearselected() 
    
    -- D. Resolution
    -- On attend que le maillage soit fini avant de continuer
    if (mi_analyze(1) == 0) then
        print("Erreur critique lors du maillage/calcul à l'etape " .. i)
        -- On continue quand même pour voir, mais c'est mauvais signe
    end
    
    mi_loadsolution() -- Ouvre la fenetre de resultats
    
    -- E. Lecture Resultats
    -- On attend un peu (hack) ou on espère que la fenêtre est active
    i_circ, v_circ, flux = mo_getcircuitproperties("Bobine")
    
    -- Securite : Si flux est nil, c'est que la lecture a echoue
    if (flux == nil) then 
        flux = 0 
        print("Attention: Lecture flux echouee (fenêtre resultats non trouvee ?)")
    end

    -- F. Ecriture
    write(handle, format("%g, %g, %g\n", t, deplacement_absolu, flux))
    print(format("Etape %d : Disp=%.3f mm -> Flux=%g", i, deplacement_absolu, flux))
    
    -- G. FERMETURE RESULTATS
    mo_close()
    
    pos_precedente = deplacement_absolu
end

closefile(handle)
print("Termine avec succès !")