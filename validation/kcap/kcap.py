"""SGLang plugin: dump each PP stage's boundary tensors for one prefill.

Every scheduler process loads it (SGLANG_PLUGINS=kcap). After load_model a
forward hook on `language_model.model` saves, for the first forward of
exactly KCAP_TOKENS tokens: the stage's input ids (first stage), the PP proxy
tensors it received (`hidden_states`, `residual` = the attention-residual
bank) and the ones it sends, or the final hidden on the last stage. Files go
to KCAP_DIR/l<start>-<end>/ as raw bytes plus meta.json.
"""
import inspect
import json
import os


def _save(d, name, t, meta):
    import torch

    t = t.detach().contiguous()
    with open(os.path.join(d, name), "wb") as f:
        f.write(t.cpu().view(-1).view(torch.uint8).numpy().tobytes())
    meta[name] = {"dtype": str(t.dtype), "shape": list(t.shape)}


def _after_load(result, runner, *args, **kwargs):
    want = int(os.environ["KCAP_TOKENS"])
    root = os.environ["KCAP_DIR"]
    lm = runner.model.language_model.model
    sig = inspect.signature(lm.forward)
    done = [False]

    def hook(mod, args, kwargs, out):
        import torch

        if done[0]:
            return
        a = sig.bind(*args, **kwargs).arguments
        ids, embeds, proxy = a.get("input_ids"), a.get("inputs_embeds"), a.get("pp_proxy_tensors")
        n = next(x.shape[0] for x in (ids, embeds, proxy and proxy["hidden_states"]) if x is not None)
        if n != want:
            return
        done[0] = True
        d = os.path.join(root, f"l{lm.start_layer}-{lm.end_layer}")
        os.makedirs(d, exist_ok=True)
        meta = {"start_layer": lm.start_layer, "end_layer": lm.end_layer, "tokens": n}
        if proxy is None:
            if ids is not None:
                _save(d, "in.input_ids", ids.to(torch.int64), meta)
            if embeds is not None:
                _save(d, "in.inputs_embeds", embeds, meta)
        else:
            for k in ("hidden_states", "residual"):
                _save(d, f"in.{k}", proxy[k], meta)
        if hasattr(out, "tensors"):
            for k in ("hidden_states", "residual"):
                _save(d, f"out.{k}", out[k], meta)
        else:
            _save(d, "out.final_hidden", out if not isinstance(out, tuple) else out[0], meta)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)

    lm.register_forward_hook(hook, with_kwargs=True)
    return result


def register():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.load_model", _after_load, HookType.AFTER
    )
