# -*- Python -*-

_TF_ACL_ROOT = "TF_ACL_ROOT"


def if_acl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with ACL.

    Returns a select statement which evaluates to if_true if we're building
    with ACL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "//third_party/acl:using_acl": if_true,
        "//conditions:default": if_false
    })


def _enable_local_acl(repository_ctx):
  return _TF_ACL_ROOT in repository_ctx.os.environ


def _acl_autoconf_impl(repository_ctx):
  """Implementation of the local_acl_autoconf repository rule."""

  # Symlink lib and include local folders.
  acl_root = repository_ctx.os.environ[_TF_ACL_ROOT]
  acl_lib_path = "%s/build" % acl_root
  repository_ctx.symlink(acl_lib_path, "lib")
  acl_include_path = "%s/include" % acl_root
  repository_ctx.symlink(acl_include_path, "include")
  acl_arm_compute_path = "%s/arm_compute" % acl_root
  repository_ctx.symlink(acl_arm_compute_path, "arm_compute")
  acl_support_path = "%s/support" % acl_root
  repository_ctx.symlink(acl_support_path, "support")
  acl_utils_path = "%s/utils" % acl_root
  repository_ctx.symlink(acl_utils_path, "utils")

#  acl_license_path = "%s/LICENSE" % acl_root
#  repository_ctx.symlink(mkl_license_path, "LICENSE")

  # Also setup BUILD file.
  repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")


acl_repository = repository_rule(
      implementation = _acl_autoconf_impl,
      environ = [
          _TF_ACL_ROOT,
      ],
      attrs = {
          "build_file": attr.label(),
          "repository": attr.string(),
          "urls": attr.string_list(default = []),
          "sha256": attr.string(default = ""),
          "strip_prefix": attr.string(default = ""),
      },
  )
