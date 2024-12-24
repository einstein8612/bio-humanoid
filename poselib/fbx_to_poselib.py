# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
from poselib.skeleton.skeleton3d import SkeletonMotion
from tqdm import tqdm
from multiprocessing import Pool, Lock, Value

# Define folders
cmu_folder = "data/cmu"
motion_folder = "data/motions"
files = os.listdir(cmu_folder)

# Shared counter and lock for progress bar
counter = Value('i', 0)
lock = Lock()

def _process(file_name: str):
    """Processes a single FBX file to generate a SkeletonMotion file."""
    fbx_file = os.path.join(cmu_folder, file_name)
    motion_file = os.path.join(motion_folder, file_name[:-4]+".npy")
    
    # Skip existing
    if os.path.exists(motion_file):
        return

    # Import FBX file and process motion
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file,
        root_joint="Hips",
        fps=60
    )
    motion.to_file(motion_file)

    # Update the progress bar
    with lock:
        counter.value += 1

# Ensure output folder exists
os.makedirs(motion_folder, exist_ok=True)

# Initialize progress bar
with tqdm(desc="Processing files", total=len(files)) as pbar:
    def update_progress(*args):
        with lock:
            pbar.update(1)

    # Process files with multiprocessing
    with Pool(processes=os.cpu_count()) as pool:
        for file_name in files:
            pool.apply_async(_process, args=(file_name,), callback=update_progress)
        pool.close()
        pool.join()