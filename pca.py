import torch

from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.evals import make_pile_data_loader
from datasets import load_dataset


def get_activations(layer, device='cpu'):
    model = HookedTransformer.from_pretrained('gelu-4l')
    model.eval()

    loader = make_pile_data_loader(tokenizer=model.tokenizer, batch_size=8)

    mlp_state = dict()
    def pre_hook(value, hook):
        h = value.detach().clone().cpu()
        # h is batch_size x seq_len x hidden_size. We flatten the batch and seq_len dimensions
        mlp_state['pre_act'] = h.view(-1, h.size(-1))
        return value

    def post_hook(value, hook):
        h = value.detach().clone().cpu()
        mlp_state['post_act'] = h.view(-1, h.size(-1))
        return value

    hooks = [
        (f'blocks.{layer}.mlp.hook_pre', pre_hook),
        (f'blocks.{layer}.mlp.hook_post', post_hook),
        ]

    pre_acts = torch.zeros((1, model.cfg.d_mlp))
    post_acts = torch.zeros((1, model.cfg.d_mlp))

    for idx, batch in tqdm(enumerate(loader)):
        if idx > 10:
            break
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            model(batch['tokens'].to(device))

            # extend pre and post acts with the current hook state
            pre_acts = torch.cat([pre_acts, mlp_state['pre_act']], dim=0)
            post_acts = torch.cat([post_acts, mlp_state['post_act']], dim=0)

    # Remove the first empty row created by initialization
    pre_acts = pre_acts[1:]
    post_acts = post_acts[1:]

    ### pca ###

    # Standardize the activations
    pre_acts = (pre_acts - pre_acts.mean(dim=0)) / pre_acts.std(dim=0)
    post_acts = (post_acts - post_acts.mean(dim=0)) / post_acts.std(dim=0)

    # Compute the covariance matrix (choose pre_acts or post_acts based on your needs)
    pre_cov_matrix = torch.mm(pre_acts.T, pre_acts) / pre_acts.shape[0]
    post_cov_matrix = torch.mm(post_acts.T, post_acts) / post_acts.shape[0]

    # Singular Value Decomposition
    U_pre, S_pre, V_pre = torch.svd(pre_cov_matrix)
    U_post, S_post, V_post = torch.svd(post_cov_matrix)

    print(f'pre_acts shape: {pre_acts.shape}')
    print(f'pre_cov_matrix shape: {pre_cov_matrix.shape}')
    print(f'U_pre shape: {U_pre.shape}')
    print(f'S_pre shape: {S_pre.shape}')
    print(f'V_pre shape: {V_pre.shape}')



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_activations(layer=0, device=device)


