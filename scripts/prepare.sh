# CREPE
negative_types=(swap negate atom)    
complexities=(4 5 6 7 8 9 10 11 12)
for negative_type in "${negative_types[@]}"
do
    for complexity in "${complexities[@]}"
    do
        python prepare.py --is_crepe --crepe_negative_type $negative_type --crepe_complexity $complexity
    done
done

# SUGARCREPE
negative_types=(add_att add_obj replace_att replace_obj replace_rel swap_att swap_obj)

for negative_type in "${negative_types[@]}"
do
    python prepare.py --is_sugarCrepe --sugarCrepe_negative_type $negative_type 
done

python prepare.py --is_svo
python prepare.py --is_aro
