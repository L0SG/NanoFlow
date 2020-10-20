def build_model(args_dict):
    from models import waveflow, nanoflow_naive, nanoflow_decomp, nanoflow

    # change dict to dot notation for code re-use
    # https://stackoverflow.com/questions/16279212/how-to-use-dot-notation-for-dict-in-python
    from types import SimpleNamespace
    for key, val in args_dict.items():
        if val == 'true' or val == 'True':
            args_dict[key] = True
        elif val == 'false' or val == 'False':
            args_dict[key] = False
    args = SimpleNamespace(**args_dict)

    if args.coupling_type == 'waveflow':
        print('loading WaveFlow model...')
        model = waveflow.WaveFlow(in_channel=1,
                                  cin_channel=args.cin_channels,
                                  res_channel=args.res_channels,
                                  n_height=args.n_height,
                                  n_flow=args.n_flow,
                                  n_layer=args.n_layer,
                                  layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                                  coupling_type='affine',
                                  )
    elif args.coupling_type == 'nanoflow_naive':
        print('loading NanoFlowNaive model...')
        model = nanoflow_naive.NanoFlowNaive(in_channel=1,
                                             cin_channel=args.cin_channels,
                                             res_channel=args.res_channels,
                                             n_height=args.n_height,
                                             n_flow=args.n_flow,
                                             n_layer=args.n_layer,
                                             layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                                             )
    elif args.coupling_type == 'nanoflow_decomp':
        print('loading NanoFlowDecomp model...')
        model = nanoflow_decomp.NanoFlowDecomp(in_channel=1,
                                               cin_channel=args.cin_channels,
                                               res_channel=args.res_channels,
                                               n_height=args.n_height,
                                               n_flow=args.n_flow,
                                               n_layer=args.n_layer,
                                               layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                                               coupling_type='affine'
                                               )
    elif args.coupling_type == 'nanoflow':
        print('loading NanoFlow model...')
        model = nanoflow.NanoFlow(in_channel=1,
                                  cin_channel=args.cin_channels,
                                  res_channel=args.res_channels,
                                  n_height=args.n_height,
                                  n_flow=args.n_flow,
                                  n_layer=args.n_layer,
                                  layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                                  size_flow_embed=args.size_flow_embed,
                                  coupling_type='affine',
                                  use_weightnorm_embed=args.use_weightnorm_embed,
                                  )
    else:
        raise NotImplementedError("unknown coupling_type. check the config file!")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model built: number of parameters: {}".format(total_params))
    model = model.cuda()
    return model