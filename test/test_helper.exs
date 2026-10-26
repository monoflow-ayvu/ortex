# resnet50 is opt-in, not implicitly skipped: an absent model used to exclude the tag
# silently, so a full run could look green while the only real end-to-end CPU model never
# executed. Its export is a manual step (python/export_resnet.py) and its weights are not
# in the repo.
exclude =
  if System.get_env("ORTEX_RESNET50") in ["1", "true"] do
    []
  else
    [:resnet50]
  end

ExUnit.start(exclude: exclude)
