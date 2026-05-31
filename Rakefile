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
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
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
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
      ] + extra_cmake_args("CPPADCG_CMAKE_ARGS")
    },
    env: lambda {
      { "CPPAD_HOME" => prefix_dir }
    }
  }
}.freeze

def prefix_dir
  File.expand_path(ENV.fetch("AUTODIFF_PREFIX", DEFAULT_PREFIX), ROOTDIR)
end

def build_root
  File.expand_path(ENV.fetch("AUTODIFF_BUILDDIR", DEFAULT_BUILDDIR), ROOTDIR)
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
  source_dir = component_source(name)
  cmake_file = File.join(source_dir, "CMakeLists.txt")

  raise RuntimeError, "Missing source directory #{source_dir}".red unless Dir.exist?(source_dir)
  raise RuntimeError, "Missing #{cmake_file}".red unless File.exist?(cmake_file)
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
    CPPAD_CMAKE_ARGS
    CPPADCG_CMAKE_ARGS
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
    ["parallel", PARALLEL.strip.empty? ? "disabled" : PARALLEL.strip],
    ["prefix", prefix_dir],
    ["build_root", build_root]
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

def install_component(name)
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

desc "show current configuration"
task :info do
  print_configuration
end

desc "configure and install CppAd into ./lib or AUTODIFF_PREFIX"
task :cppad do
  print_configuration
  FileUtils.mkdir_p prefix_dir
  build_component(:cppad)
end

desc "configure and install CppAdCodegen into ./lib or AUTODIFF_PREFIX"
task :cppadcg => :cppad do
  build_component(:cppadcg)
end

desc "build and install CppAd and CppAdCodegen into ./lib"
task :build_common => :cppadcg do
end

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

desc "remove CppAdCodegen build directory"
task :clean_cppadcg do
  FileUtils.rm_rf component_build_dir(:cppadcg)
end

task :clean_osx   => :clean do end
task :clean_linux => :clean do end
task :clean_mingw => :clean do end
task :clean_win   => :clean do end
