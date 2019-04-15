
##pinn2json bugs

1. roi : 6600 (VMAT) parse error: roi:6600;(VMAT)
  pinn2Json.py-> pinnProcess
    linenum:112
    # remove roi_name like : name : 6000 (Trial_1)
        pinnFileText = re.sub("name:\s(\d+)\s\((.*)\)",
                              'name:ring', pinnFileText)

2. roi: pinn2json.py, curveList modifying
  1. pinn2json.py-> read
    linenum:255-2016
    # modifiying ROI-curveList 2019-04-12
        pinnFileText = re.sub('\n\"num_curve\"(.*)\n',
                              '\n\"num_curve\" : [\n', pinnFileText)
        pinnFileText = re.sub('\n\"surface_mesh\"',
                              '\n],\n\"surface_mesh\"', pinnFileText)
        pinnFileText = re.sub('\n\"curve\"\s:\s\{\n', "\n{", pinnFileText)

3. pinn2Json, failed to read,
  1. Plan.Pinnacle.Machine
  2. plan.OrbitConstraints
  3. plan.OrbitOjectives
