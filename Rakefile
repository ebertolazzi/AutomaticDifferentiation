require "shellwords"

if File.exist?(File.expand_path("./cmake_utils/Rakefile_common.rb", File.dirname(__FILE__)))
  require_relative "./cmake_utils/Rakefile_common.rb"
else
  require_relative "../Rakefile_common.rb"
end

ROOTDIR         = File.expand_path(__dir__)
DEFAULT_PREFIX  = File.join(ROOTDIR, "lib")
DEFAULT_BUILDDIR = File.join(ROOTDIR, "build")
DEFAULT_GENERATOR = ENV.fetch("CMAKE_GENERATOR", "Ninja")
DEFAULT_CMAKE     = ENV.fetch("CMAKE", "cmake")
DEFAULT_CXX_STANDARD = ENV.fetch("AUTODIFF_CXX_STANDARD", "17")

COMPONENTS = {
  eigen5: {
    label: "Eigen5",
    source: File.join(ROOTDIR, "submodules", "Eigen5"),
    build:  "Eigen5",
    cmake_args: lambda {
      [
        "-DCMAKE_INSTALL_PREFIX=#{prefix_dir}",
        "-DCMAKE_CXX_STANDARD=#{DEFAULT_CXX_STANDARD}",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DBUILD_TESTING=OFF",
        "-DEIGEN_BUILD_TESTING=OFF",
        "-DEIGEN_BUILD_DOC=OFF",
        "-DEIGEN_BUILD_DEMOS=OFF",
        "-DEIGEN_BUILD_BLAS=OFF",
        "-DEIGEN_BUILD_LAPACK=OFF",
        "-DEIGEN_BUILD_PKGCONFIG=ON",
        "-DEIGEN_BUILD_CMAKE_PACKAGE=ON"
      ] + extra_cmake_args("EIGEN5_CMAKE_ARGS")
    }
  },
  cppad: {
    label: "CppAd",
    source: File.join(ROOTDIR, "submodules", "CppAd"),
    build:  "CppAd",
    cmake_args: lambda {
      [
        "-Dcppad_prefix=#{prefix_dir}",
        "-Dcmake_install_includedirs=include",
        "-Dcmake_install_libdirs=lib",
        "-Dcppad_static_lib=#{cppad_static_lib}",
        "-DCMAKE_CXX_STANDARD=#{DEFAULT_CXX_STANDARD}",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DCMAKE_PREFIX_PATH=#{cmake_prefix_path}"
      ] + extra_cmake_args("CPPAD_CMAKE_ARGS")
    }
  },
  cppadcg: {
    label: "CppAdCodegen",
    source: File.join(ROOTDIR, "submodules", "CppAdCodegen"),
    build:  "CppAdCodegen",
    cmake_args: lambda {
      [
        "-DCMAKE_INSTALL_PREFIX=#{prefix_dir}",
        "-DCPPAD_HOME=#{prefix_dir}",
        "-DCMAKE_CXX_STANDARD=#{DEFAULT_CXX_STANDARD}",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DCMAKE_PREFIX_PATH=#{cmake_prefix_path}"
      ] + extra_cmake_args("CPPADCG_CMAKE_ARGS")
    },
    env: lambda {
      {
        "CPPAD_HOME" => prefix_dir,
        "CMAKE_PREFIX_PATH" => cmake_prefix_path
      }
    }
  },
  tinyad: {
    label: "TinyAD",
    source: File.join(ROOTDIR, "submodules", "TinyAD"),
    build:  "TinyAD",
    kind:   :header_only,
    cmake_check: true,
    include_dir: File.join(ROOTDIR, "submodules", "TinyAD", "include"),
    cmake_args: lambda {
      [
        "-DCMAKE_CXX_STANDARD=#{DEFAULT_CXX_STANDARD}",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DCMAKE_PREFIX_PATH=#{cmake_prefix_path}",
        "-DEigen3_DIR=#{eigen3_cmake_dir}",
        "-DTINYAD_UNIT_TESTS=OFF",
        "-DBUILD_TESTING=OFF"
      ] + extra_cmake_args("TINYAD_CMAKE_ARGS")
    }
  }
}.freeze

def prefix_dir
  File.expand_path(ENV.fetch("AUTODIFF_PREFIX", DEFAULT_PREFIX), ROOTDIR)
end

def build_root
  File.expand_path(ENV.fetch("AUTODIFF_BUILDDIR", DEFAULT_BUILDDIR), ROOTDIR)
end

def tests_source_dir
  File.join(ROOTDIR, "tests")
end

def tests_build_dir
  File.join(build_root, "tests")
end

def tests_codegen_gradient_dir
  File.join(tests_source_dir, "cppadcg_benchmark_runtime")
end

def tests_codegen_hessian_dir
  File.join(tests_source_dir, "cppadcg_hessian_runtime")
end

def cmake_prefix_path
  [prefix_dir, ENV.fetch("CMAKE_PREFIX_PATH", nil)].compact.reject(&:empty?).join(";")
end

def eigen3_cmake_dir
  File.join(prefix_dir, "share", "eigen3", "cmake")
end

def component_source(name)
  COMPONENTS.fetch(name)[:source]
end

def component_build_dir(name)
  File.join(build_root, COMPONENTS.fetch(name)[:build])
end

def build_type
  COMPILE_DEBUG ? "Debug" : "Release"
end

def cppad_static_lib
  COMPILE_DYNAMIC ? "FALSE" : "TRUE"
end

def extra_cmake_args(env_name)
  shared = Shellwords.split(ENV.fetch("CMAKE_ARGS", ""))
  specific = Shellwords.split(ENV.fetch(env_name, ""))
  shared + specific
end

def shell_join(argv)
  Shellwords.join(argv)
end

def env_shell_prefix(env_hash)
  return "" if env_hash.nil? || env_hash.empty?

  env_hash.map { |key, value| "#{key}=#{Shellwords.escape(value)}" }.join(" ") + " "
end

def run_command(argv, env = {})
  sh env_shell_prefix(env) + shell_join(argv)
end

def ensure_command!(command)
  return if system("command -v #{Shellwords.escape(command)} >/dev/null 2>&1")

  raise RuntimeError, "Cannot find required command `#{command}`".red
end

def ensure_component!(name)
  spec = COMPONENTS.fetch(name)
  source_dir = component_source(name)
  cmake_file = File.join(source_dir, "CMakeLists.txt")

  raise RuntimeError, "Missing source directory #{source_dir}".red unless Dir.exist?(source_dir)
  if spec[:kind] == :header_only
    include_dir = spec[:include_dir]
    raise RuntimeError, "Missing include directory #{include_dir}".red unless Dir.exist?(include_dir)
    if spec[:cmake_check]
      raise RuntimeError, "Missing #{cmake_file}".red unless File.exist?(cmake_file)
    end
  else
    raise RuntimeError, "Missing #{cmake_file}".red unless File.exist?(cmake_file)
  end
end

def component_needs_configure?(name)
  spec = COMPONENTS.fetch(name)
  spec[:kind] != :header_only || spec[:cmake_check]
end

def component_status(name)
  Dir.exist?(component_source(name)) ? "ok" : "missing"
end

def environment_overrides
  %w[
    AUTODIFF_PREFIX
    AUTODIFF_BUILDDIR
    AUTODIFF_CXX_STANDARD
    CMAKE
    CMAKE_GENERATOR
    CMAKE_ARGS
    EIGEN5_CMAKE_ARGS
    CPPAD_CMAKE_ARGS
    CPPADCG_CMAKE_ARGS
    TINYAD_CMAKE_ARGS
    AUTODIFF_TEST_CMAKE_ARGS
    AUTODIFF_BENCH_ARGS
  ].filter_map do |key|
    next unless ENV.key?(key)

    [key, ENV.fetch(key, "")]
  end
end

def print_table(title, rows)
  key_width   = rows.map { |key, _| key.length }.max || 0
  value_width = rows.map { |_, value| value.length }.max || 0
  inner_width = [title.length, key_width + value_width + 5].max
  border      = "+-#{'-' * inner_width}-+"

  puts border.green
  puts "| #{title.ljust(inner_width)} |".green
  puts border.green
  rows.each do |key, value|
    line = "#{key.ljust(key_width)} : #{value}"
    puts "| #{line.ljust(inner_width)} |".green
  end
  puts border.green
end

def print_configuration
  config_rows = [
    ["build_type", build_type],
    ["compile_shared", COMPILE_DYNAMIC.to_s],
    ["generator", DEFAULT_GENERATOR],
    ["cmake", DEFAULT_CMAKE],
    ["cxx_standard", DEFAULT_CXX_STANDARD],
    ["eigen3_dir", eigen3_cmake_dir],
    ["cmake_prefix_path", cmake_prefix_path],
    ["parallel", PARALLEL.strip.empty? ? "disabled" : PARALLEL.strip],
    ["prefix", prefix_dir],
    ["build_root", build_root],
    ["tests_source", tests_source_dir],
    ["tests_build", tests_build_dir]
  ]

  component_rows = COMPONENTS.map do |name, spec|
    ["#{spec[:label]}", "#{component_status(name)} | #{component_source(name)}"]
  end

  print_table("AutomaticDifferentiation", config_rows)
  print_table("Components", component_rows)

  overrides = environment_overrides
  unless overrides.empty?
    print_table("Environment Overrides", overrides)
  end
end

def configure_component(name)
  ensure_component!(name)
  return unless component_needs_configure?(name)

  ensure_command!(DEFAULT_CMAKE)

  source_dir = component_source(name)
  build_dir  = component_build_dir(name)
  args       = COMPONENTS.fetch(name)[:cmake_args].call
  env        = COMPONENTS.fetch(name).fetch(:env, -> { {} }).call

  FileUtils.mkdir_p build_dir

  puts "Configure #{COMPONENTS.fetch(name)[:label]}".green
  run_command(
    [
      DEFAULT_CMAKE,
      "-S", source_dir,
      "-B", build_dir,
      "-G", DEFAULT_GENERATOR,
      "-DCMAKE_BUILD_TYPE=#{build_type}"
    ] + args,
    env
  )
end

def install_header_only_component(name)
  spec        = COMPONENTS.fetch(name)
  include_dir = spec[:include_dir]
  dest_dir    = File.join(prefix_dir, "include")

  raise RuntimeError, "Include directory #{include_dir} does not exist".red unless Dir.exist?(include_dir)

  FileUtils.mkdir_p dest_dir

  puts "Install #{spec[:label]}".green
  FileUtils.cp_r File.join(include_dir, "."), dest_dir
end

def install_component(name)
  if COMPONENTS.fetch(name)[:kind] == :header_only
    install_header_only_component(name)
    return
  end

  build_dir = component_build_dir(name)
  env       = COMPONENTS.fetch(name).fetch(:env, -> { {} }).call

  raise RuntimeError, "Build directory #{build_dir} does not exist".red unless Dir.exist?(build_dir)

  puts "Install #{COMPONENTS.fetch(name)[:label]}".green
  run_command(
    [
      DEFAULT_CMAKE,
      "--build", build_dir,
      "--config", build_type,
      "--target", "install"
    ] + Shellwords.split(PARALLEL),
    env
  )
end

def build_component(name)
  configure_component(name)
  install_component(name)
end

def ensure_tests_project!
  cmake_file = File.join(tests_source_dir, "CMakeLists.txt")

  raise RuntimeError, "Missing tests directory #{tests_source_dir}".red unless Dir.exist?(tests_source_dir)
  raise RuntimeError, "Missing #{cmake_file}".red unless File.exist?(cmake_file)
end

def configure_tests_project
  ensure_tests_project!
  ensure_command!(DEFAULT_CMAKE)

  FileUtils.mkdir_p tests_build_dir

  puts "Configure tests".green
  run_command(
    [
      DEFAULT_CMAKE,
      "-S", tests_source_dir,
      "-B", tests_build_dir,
      "-G", DEFAULT_GENERATOR,
      "-DCMAKE_BUILD_TYPE=#{build_type}",
      "-DAUTODIFF_PREFIX=#{prefix_dir}",
      "-DCMAKE_PREFIX_PATH=#{cmake_prefix_path}",
      "-DEigen3_DIR=#{eigen3_cmake_dir}"
    ] + extra_cmake_args("AUTODIFF_TEST_CMAKE_ARGS")
  )
end

def build_tests_project
  configure_tests_project

  puts "Build tests".green
  run_command(
    [
      DEFAULT_CMAKE,
      "--build", tests_build_dir,
      "--config", build_type
    ] + Shellwords.split(PARALLEL)
  )
end

def tests_executable_candidates
  tests_executable_candidates_for("autodiff_gradient_benchmark")
end

def tests_executable_candidates_for(base_name)
  exe = Gem.win_platform? ? "#{base_name}.exe" : base_name

  [
    File.join(tests_build_dir, exe),
    File.join(tests_build_dir, build_type, exe)
  ]
end

def tests_executable_path
  tests_executable_candidates.find { |path| File.exist?(path) } || tests_executable_candidates.first
end

def tests_executable_path_for(base_name)
  candidates = tests_executable_candidates_for(base_name)
  candidates.find { |path| File.exist?(path) } || candidates.first
end

def run_tests_project
  build_tests_project

  puts "Run tests".green
  run_command(
    [
      DEFAULT_CMAKE,
      "--test-dir", tests_build_dir,
      "--build-config", build_type,
      "--output-on-failure"
    ]
  )
end

def run_benchmark
  run_benchmark_for("autodiff_gradient_benchmark", "gradient benchmark")
end

def run_hessian_benchmark
  run_benchmark_for("autodiff_hessian_benchmark", "hessian benchmark")
end

def run_benchmark_for(base_name, label)
  build_tests_project

  exe_path = tests_executable_path_for(base_name)
  raise RuntimeError, "Cannot find #{label} executable #{exe_path}".red unless File.exist?(exe_path)

  puts "Run #{label}".green
  run_command([exe_path] + Shellwords.split(ENV.fetch("AUTODIFF_BENCH_ARGS", "")))
end

desc "show current configuration"
task :info do
  print_configuration
end

desc "configure and install Eigen5 into ./lib or AUTODIFF_PREFIX"
task :eigen5 do
  print_configuration
  FileUtils.mkdir_p prefix_dir
  build_component(:eigen5)
end

desc "configure and install CppAd into ./lib or AUTODIFF_PREFIX"
task :cppad => :eigen5 do
  build_component(:cppad)
end

desc "configure and install CppAdCodegen into ./lib or AUTODIFF_PREFIX"
task :cppadcg => :cppad do
  build_component(:cppadcg)
end

desc "configure TinyAD against local Eigen5 without building tests"
task :tinyad_check => :eigen5 do
  FileUtils.mkdir_p prefix_dir
  configure_component(:tinyad)
end

desc "configure TinyAD against local Eigen5 and install headers into ./lib/include"
task :tinyad => :tinyad_check do
  FileUtils.mkdir_p prefix_dir
  install_component(:tinyad)
end

desc "build and install Eigen5, CppAd, CppAdCodegen, and TinyAD into ./lib"
task :build_common => [:cppadcg, :tinyad] do
end

desc "configure the local tests project against ./lib"
task :tests_configure => :build_common do
  configure_tests_project
end

desc "build the local tests project"
task :tests_build => :build_common do
  build_tests_project
end

desc "run the local gradient and hessian tests with ctest"
task :tests => :build_common do
  run_tests_project
end

desc "run the gradient benchmark executable directly"
task :bench_gradient => :build_common do
  run_benchmark
end

desc "run the hessian benchmark executable directly"
task :bench_hessian => :build_common do
  run_hessian_benchmark
end

desc "run both gradient and hessian benchmark executables"
task :bench => [:bench_gradient, :bench_hessian] do
end

task :test => :tests do end

desc "build and install for macOS"
task :build_osx   => :build_common do end

desc "build and install for Linux"
task :build_linux => :build_common do end

desc "build and install for MinGW"
task :build_mingw => :build_common do end

desc "build and install for Visual Studio"
task :build_win   => :build_common do end

desc "install headers and libraries into ./lib"
task :install => :build do end

task :install_osx   => :build_osx do end
task :install_linux => :build_linux do end
task :install_mingw => :build_mingw do end
task :install_win   => :build_win do end

desc "remove only CMake build directories"
task :clean_build do
  FileUtils.rm_rf build_root
end

desc "remove install prefix and build directories"
task :clean do
  FileUtils.rm_rf prefix_dir
  FileUtils.rm_rf build_root
end

desc "remove CppAd build directory"
task :clean_cppad do
  FileUtils.rm_rf component_build_dir(:cppad)
end

desc "remove Eigen5 build directory"
task :clean_eigen5 do
  FileUtils.rm_rf component_build_dir(:eigen5)
end

desc "remove CppAdCodegen build directory"
task :clean_cppadcg do
  FileUtils.rm_rf component_build_dir(:cppadcg)
end

desc "remove tests build directory"
task :clean_tests do
  FileUtils.rm_rf tests_build_dir
  FileUtils.rm_rf tests_codegen_gradient_dir
  FileUtils.rm_rf tests_codegen_hessian_dir
end

task :clean_osx   => :clean do end
task :clean_linux => :clean do end
task :clean_mingw => :clean do end
task :clean_win   => :clean do end
