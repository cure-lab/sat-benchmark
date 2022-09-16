from urllib.parse import urlsplit, urlunsplit
import os

# from .registry import is_model, is_model_in_modules, model_entrypoint
from .registry import is_model, model_entrypoint
from .helpers import load_checkpoint
# from .layers import set_layer_config
# from .hub import load_model_config_from_hf


# def parse_model_name(model_name):
#     model_name = model_name.replace('hf_hub', 'hf-hub')  # NOTE for backwards compat, to deprecate hf_hub use
#     parsed = urlsplit(model_name)
#     assert parsed.scheme in ('', 'timm', 'hf-hub')
#     if parsed.scheme == 'hf-hub':
#         # FIXME may use fragment as revision, currently `@` in URI path
#         return parsed.scheme, parsed.path
#     else:
#         model_name = os.path.split(parsed.path)[-1]
#         return 'timm', model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    # if remove_source:
    #     model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name,
        pretrained=False,
        pretrained_cfg=None,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # model_source, model_name = parse_model_name(model_name)
    # if model_source == 'hf-hub':
        # FIXME hf-hub source overrides any passed in pretrained_cfg, warn?
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        # pretrained_cfg, model_name = load_model_config_from_hf(model_name)

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    # with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
    #     model = create_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)
    # model = create_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)
    model = create_fn()

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
