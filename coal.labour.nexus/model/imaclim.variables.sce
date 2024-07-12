secnames = [
"coal"
"oil"
"gaz"
"Et"
"elec"
"cons"
"comp"
"air"
"mer"
"OT"
"agri"
"indu"
];


regnames = ['USA',
'CAN',
'EUR',
'JAN',
'CIS',
'CHN',
'IND',
'BRA',
'MDE',
'AFR',
'RAS',
'RAL'
];

ind_usa = 1;
ind_can = 2;
ind_eur = 3;
ind_jan = 4;
ind_cis = 5;
ind_chn = 6;
ind_ind = 7;
ind_bra = 8;
ind_mde = 9;
ind_afr = 10;
ind_ras = 11;
ind_ral = 12;

reg=12;
sec=12;

indice_coal 	= 1;
indice_oil  	= 2;
indice_gaz  	= 3;
indice_Et   	= 4;
indice_elec 	= 5;
indice_construction	= 6;
indice_composite    	= 7; // ~ services sector
indice_air  	= 8;
indice_mer  	= 9;
indice_OT   	= 10;
indice_agriculture	= 11;
indice_industries 	= 12;


base_year_simulation = 2014;
final_year_simulation=2100;
if ~isdef('TimeHorizon')
    TimeHorizon=final_year_simulation-base_year_simulation; // warning : the default time horizon is 99 but could be overwritten in STUDY.sce
end

function out=sg_get_var(matname,which_lines,which_columns,nb_lines,default_orient,which_years,forceColumn)
    //
    // out=sg_get_var(matname,[which_lines,[which_columns,[nb_lines,[default_orient,[which_years,[forceColumn]]]]]])
    //
    //destinee a remplacer les boucle avec les _temp pour recuperer les variables enregistree dans le format "une colonne= 1 annee"
    //
    //INPUTS
    // matname : nom de la variable (aves ou sans _sav), ou variable elle même
    // which_lines : lignes, cad souvent regions, à sortir par exemple 1:reg, ou 1:4, ou 3. Par defaut 1:reg
    // which_columns : colonnes (souvent secteurs) à sortir (par exemple [indice_coal:indice_elec]). Par defaut : (toutes le scolonnes)
    // nb_lines : nombre de lignes de la matrice originale. Par defaut, reg.
    // which_years : temps à sortir. par defaut toutes les années.
    // default_orient : transpose la matrice originale. Par defaut, %T
    // forceColumn : force la sortie a être une colonne quand which_years est un nombre. Par defaut, %F
    //
    //OUTPUTS
    // out : colonne (size(which_lines,"*") x size(which_columns,"*") ) x size(which_years)
    // si size(which_years) = 1 et ~forceColumn une seule année selectionnée, alors out est une matrice et pas :
    //                          out : matrice size(which_lines,"*") x size(which_columns,"*") 
    //EXEMPLES
    // norm(sg_get_var("E_reg_use")-E_reg_use_sav) //meme chose 
    // sg_get_var(E_reg_use_sav,ind_eur,iu_df,reg,%f,1:10) //les emissions des menages europeens les dix premieres années
    // sum(sg_get_var("E_reg_use",1:4,iu_df,reg,%f,1:10),"r") //les emissions des menages de l'ocde dans les dix premieres années
    // sgv("taxCO2_DF",:,:,reg,1,current_time_im) //récupère taxCO2_DF en entier à l'année i

    //PREPROCESS
    if typeof(matname)=="string" //user provided the name of the variable
        matname = stripblanks(matname,%t);
        matname = strsubst (matname,"_sav","")+"_sav" //we get sure to get varname_sav
        global(matname)
        mat = evstr( matname) 
        if isempty(mat) //varname was not loaded
            if ~isdef("metaRecMessOn")
                metaRecMessOn = %t;
            end
            message("sg_get_var ldsaves "+matname)
            ldsav(matname)
            mat = evstr( matname) 
        end    
    else
        mat = matname
    end    

    //DEFAULT VALUES
    if argn(2)<7 
        forceColumn =%F;
    end    
    if argn(2)<6 
        which_years =1:TimeHorizon+1
    end    

    if argn(2)<5 
        default_orient = %t;
    end

    if argn(2)<4 
        nb_lines = reg
    end    

    if argn(2)<3 
        which_columns=:;
    end

    if argn(2)<2 
        out = mat;
        return
    end

    //WORK    
    
    
    for isg=1:size(which_years,"*")
        mattemp=matrix(mat(:,which_years(isg)),nb_lines,-1);
        if default_orient
            out(:,isg) = matrix(mattemp(which_lines,which_columns)',-1,1);
        else
            out(:,isg) = matrix(mattemp(which_lines,which_columns),-1,1);
        end    
    end

    //no column case
    if  size(which_years,"*")==1 & ~forceColumn 
        if which_lines==: & string(which_lines)~="1"
            out = matrix(out,nb_lines,-1);
        else
            out = matrix(out,size(which_lines,'*'),-1);
        end    
    end

    if isempty(out)
        warning ( whereami ()+"out is empty")
    end

endfunction
//alias au nom plus court
sgv = sg_get_var;