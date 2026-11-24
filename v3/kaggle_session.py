"""Look at, or cancel, one Kaggle session by the name of its script.

    ~/.local/share/uv/tools/kaggle/bin/python v3/kaggle_session.py show air-v3-practice-1
    ~/.local/share/uv/tools/kaggle/bin/python v3/kaggle_session.py cancel air-v3-practice-1

The Kaggle command-line tool has no "cancel", but Kaggle's programming interface has one. It
must be run with the Python of the Kaggle tool (the first path above), which has that package.
The Kaggle user name is never printed.
"""

import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (ApiCancelKernelSessionRequest,
                                                         ApiGetKernelSessionStatusRequest)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("show", "cancel"):
        sys.exit(__doc__)
    action, name = sys.argv[1], sys.argv[2]
    user = (Path.home() / ".kaggle" / "username_for_scripts").read_text().strip()
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiGetKernelSessionStatusRequest()
        request.user_name = user
        request.kernel_slug = name
        status = kaggle.kernels.kernels_api_client.get_kernel_session_status(request)
        session_id = getattr(status, "kernel_session_id", None)
        print(f"{name}: status {status.status}, session number {session_id}")
        if action == "cancel":
            if not session_id:
                sys.exit("No session number was given by Kaggle, so nothing was cancelled.")
            cancel = ApiCancelKernelSessionRequest()
            cancel.kernel_session_id = int(session_id)
            answer = kaggle.kernels.kernels_api_client.cancel_kernel_session(cancel)
            print("cancel sent; Kaggle's answer:", (getattr(answer, "error_message", "") or "accepted"))


if __name__ == "__main__":
    main()
