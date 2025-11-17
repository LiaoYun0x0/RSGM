from models.model_m import MatchingNet

def build_model(args):
    return MatchingNet(
        d_coarse_model=args.d_coarse_model,
        d_fine_model=args.d_fine_model,
        n_coarse_layers=args.n_coarse_layers,
        n_fine_layers=args.n_fine_layers,
        n_heads=args.n_heads,
        backbone_name=args.backbone_name,
        # matching_name=args.matching_name,
        match_threshold=args.match_threshold,
        window=args.window_size,
        border=args.border,
        sinkhorn_iterations=args.sinkhorn_iterations
    )
