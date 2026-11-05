# SPDX-License-Identifier: MIT
# Target only measured post-RCX drivers. Public LibreLane helpers handle the
# database, constraints, legal placement, routing guides, and output views.
source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl
read_current_odb
set_propagated_clock [all_clocks]
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl

set block [ord::get_db_block]
set units [$block getDbUnitsPerMicron]
set count 0
foreach driver_pin $::env(TINYLPU_ECO_DRIVER_PINS) {
    set separator [string last / $driver_pin]
    set instance_name [string range $driver_pin 0 [expr {$separator-1}]]
    set port_name [string range $driver_pin [expr {$separator+1}] end]
    set instance [$block findInst $instance_name]
    if {$instance eq "NULL"} {error "ECO driver instance not found: $driver_pin"}
    set driver [$instance findITerm $port_name]
    if {$driver eq "NULL" || [[$driver getMTerm] getIoType] ne "OUTPUT"} {
        error "ECO target is not an output: $driver_pin"
    }
    set net [$driver getNet]
    if {$net eq "NULL" || [llength [$net getBTerms]]} {error "ECO expects an internal net: $driver_pin"}
    set loads [list]
    foreach terminal [$net getITerms] {
        if {$terminal eq $driver} {continue}
        if {[[$terminal getMTerm] getIoType] eq "OUTPUT"} {error "Multiple drivers on ECO net"}
        lappend loads "[[$terminal getInst] getName]/[[$terminal getMTerm] getName]"
    }
    if {![llength $loads]} {error "ECO net has no loads: $driver_pin"}
    set loads [lsort $loads]
    set bbox [$instance getBBox]
    set location [list [expr {double([$bbox xMax])/$units}] [expr {double([$bbox yMin])/$units}]]
    set group_size $::env(TINYLPU_ECO_MAX_LOADS)
    for {set offset 0} {$offset<[llength $loads]} {incr offset $group_size} {
        set group [lrange $loads $offset [expr {$offset+$group_size-1}]]
        puts "TINYLPU_ECO driver=$driver_pin loads=$group cell=$::env(TINYLPU_ECO_BUFFER_CELL) location=$location"
        insert_buffer -buffer_cell $::env(TINYLPU_ECO_BUFFER_CELL) -load_pins $group \
            -location $location -buffer_name tinylpu_eco_$count -net_name tinylpu_eco_net_$count
        incr count
    }
}
global_connect
source $::env(SCRIPTS_DIR)/openroad/common/dpl.tcl
source $::env(SCRIPTS_DIR)/openroad/common/grt.tcl
estimate_parasitics -global_routing
puts "TINYLPU_ECO inserted_buffers=$count"
write_views
