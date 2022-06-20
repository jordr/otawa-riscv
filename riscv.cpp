/*
 *	otawa-riscv -- OTAWA loader for RISC-V instruction set
 *
 *	This file is part of OTAWA
 *	Copyright (c) 2017, IRIT UPS.
 *
 *	OTAWA is free software; you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation; either version 2 of the License, or
 *	(at your option) any later version.
 *
 *	OTAWA is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with OTAWA; if not, write to the Free Software
 *	Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <otawa/prog/Loader.h>
#include <otawa/proc/ProcessorPlugin.h>
#include <otawa/hard.h>
#include <otawa/program.h>
#include <gel++.h>
#include <gel++/DebugLine.h>
#include <otawa/prog/sem.h>
#include <otawa/prog/Loader.h>
#include <elm/stree/MarkerBuilder.h>

extern "C" {
#	include <riscv/grt.h>
#	include <riscv/api.h>
#	include <riscv/config.h>
}

namespace otawa { namespace riscv {

// Register description
hard::RegBank R(hard::RegBank::Make("GPR").gen(32, hard::Register::Make("r%d")));
hard::Register PC("PC", hard::Register::ADDR, 32);
hard::RegBank MISC(hard::RegBank::Make("Misc").add(PC));
static const hard::RegBank *banks_tab[] = { &R, &MISC };
static Array<const hard::RegBank *> banks_table(2, banks_tab);

} }	// otawa::riscv

#include "otawa_kind.h"
#include "otawa_target.h"
#include "otawa_delayed.h"
#include "otawa_read.h"
#include "otawa_write.h"

namespace otawa { namespace riscv {

class Process;

// Platform description
class Platform: public hard::Platform {
public:
	static const Identification ID;

	Platform(const PropList& props = PropList::EMPTY): hard::Platform(ID, props)
		{ setBanks(banks_table); }
	Platform(const Platform& platform, const PropList& props = PropList::EMPTY)
		: hard::Platform(platform, props)
		{ setBanks(banks_table); }

	// otawa::Platform overload
	virtual bool accept(const Identification& id) { return id.abi() == "eabi" && id.architecture() == "riscv"; }
	virtual const hard::Register *getSP(void) const { return R[29]; }
};
const Platform::Identification Platform::ID("riscv-eabi-");


// Inst class
class Inst: public otawa::Inst {
public:

	inline Inst(Process& process, kind_t kind, Address addr)
		: proc(process), _kind(kind), _addr(addr.offset()) {
		}
	Process &process(void) { return proc; }

	// Inst overload
	kind_t kind(void) override { return _kind; }
	address_t address(void) const override { return _addr; }
	t::uint32 size(void) const override { return 4; }

	void dump(io::Output& out) override;
	void semInsts(sem::Block &block) override;
	void readRegSet (RegSet &set) override;
	void writeRegSet(RegSet &set) override;

	// deprecated
	const Array<hard::Register *>& readRegs(void) override;
	const Array<hard::Register *>& writtenRegs(void) override;

protected:
	void decodeRegs(void);
	Process &proc;
	kind_t _kind;
	riscv_address_t _addr;
};


// BranchInst class
class BranchInst: public Inst {
public:

	inline BranchInst(Process& process, kind_t kind, Address addr)
		: Inst(process, kind, addr), _target(0), isTargetDone(false)
		{ }

	otawa::Inst *target(void) override;
	int delaySlots(void) override { return 1; }
	delayed_t delayType(void) override;

private:
	otawa::Inst *_target;
	bool isTargetDone;
};


// Segment class
class Process;
class Segment: public otawa::Segment {
public:
	Segment(Process& process,
		CString name,
		address_t address,
		t::uint32 size,
		int flags = EXECUTABLE)
	: otawa::Segment(name, address, size, flags), proc(process) { }

protected:
	otawa::Inst *decode(address_t address) override;

private:
	Process& proc;
};


class Process: public otawa::Process {
public:

	Process(Manager *manager, hard::Platform *pf, const PropList& props = PropList::EMPTY)
	:	otawa::Process(manager, props),
	 	_start(0),
	 	oplatform(pf),
	 	_platform(0),
		_memory(0),
		_decoder(0),
		map(nullptr),
		_file(nullptr),
		argc(0),
		no_stack(true),
		init(false)
	{
		ASSERTP(manager, "manager required");
		ASSERTP(pf, "platform required");

		// initialize RISC-V decoding
		_platform = riscv_new_platform();
		ASSERTP(_platform, "cannot create an arm_platform");
		_decoder = riscv_new_decoder(_platform);
		ASSERTP(_decoder, "cannot create an arm_decoder");
		_memory = riscv_get_memory(_platform, RISCV_MAIN_MEMORY);
		ASSERTP(_memory, "cannot get main arm_memory");
		riscv_lock_platform(_platform);

		// build arguments
		static char no_name[1] = { 0 };
		static char *default_argv[] = { no_name };
		char **av;
		argc = ARGC(props);
		if (argc < 0) {
			argc = 1;
			av = default_argv;
		}
		else {
			no_stack = false;
			av = ARGV(props);
		}
		argv.tie(argc, new cstring[argc]);
		for(int i = 0; i < argc; i++)
			argv[i] = av[i];
		
		// build environment
		char **ev = ENVP(props);
		if (ev != nullptr) {
			no_stack = false;
			int ec = 0;
			while(ev[ec] != nullptr) ec++;
			envp.tie(ec, new cstring[ec]);
			for(int i = 0; i < ec; i++)
				envp[i] = ev[i];
		}

		// handle features
		provide(MEMORY_ACCESS_FEATURE);
		provide(SOURCE_LINE_FEATURE);
		provide(CONTROL_DECODING_FEATURE);
		provide(REGISTER_USAGE_FEATURE);
		provide(MEMORY_ACCESSES);
		provide(otawa::DELAYED2_FEATURE);
	}

	~Process() {
		riscv_delete_decoder(_decoder);
		riscv_unlock_platform(_platform);
		if(_file != nullptr)
			delete _file;
	}

	// Process overloads
	int instSize(void) const  override { return 4; }
	hard::Platform *platform()  override { return oplatform; }
	otawa::Inst *start()  override { return _start; }

	File *loadFile(elm::CString path)  override {

		// check if there is not an already opened file !
		if(program())
			throw LoadException("loader cannot open multiple files !");

		// make the file
		File *file = new otawa::File(path);
		addFile(file);

		// build the environment
		
		/* !!TODO!!
		gel_env_t genv = *gel_default_env();
		genv.argv = argv;
		genv.envp = envp;*/
		gel::Parameter params;
		params.stack_alloc = !no_stack;
		if(argv)
			params.arg = argv;
		
		// build the GEL image
		gel::Image *gimage = nullptr;
		try {
			_file = gel::Manager::open(path);
			gimage = _file->make();
		}
		catch(gel::Exception& e) {
			throw LoadException(_ << "cannot load \"" << path
				<< "\": " << e.message());
		}

		// parse the segments
		for(auto seg: gimage->segments()) {

			// add to GLISS memory
			riscv_mem_write(_memory,
				seg->loadAddress(),
				seg->buffer().bytes(),
				seg->size());			
			
			// add the segment
			int flags = 0;
			if(seg->isExecutable())
				flags |= Segment::EXECUTABLE;
			if(seg->isWritable())
				flags |= Segment::WRITABLE;
			if(seg->isReadable())
				flags |= Segment::INITIALIZED;
			Segment *oseg = new Segment(*this,
				seg->name(),
				seg->baseAddress(),
				seg->size(),
				flags);
			file->addSegment(oseg);
		}

		// cleanup image
		delete gimage;

		// Initialize symbols
		auto symtab = _file->symbols();
		for(auto sym: symtab) {

			// compute kind and mask
			Symbol::kind_t kind = Symbol::NONE;
			t::uint32 mask = 0xffffffff;
			switch(sym->type()) {
			case gel::Symbol::FUNC:
				kind = Symbol::FUNCTION;
				mask = 0xfffffffe;
				break;
			case gel::Symbol::DATA:
				kind = Symbol::DATA;
				break;
			case gel::Symbol::OTHER_TYPE:
				kind = Symbol::LABEL;
				break;
			default:
				continue;
			}

			// build the label if required
			Symbol *osym = new Symbol(
				*file,
				sym->name(),
				kind,
				sym->value() & mask,
				sym->size());
			file->addSymbol(osym);
		}
		

		// get the entry instructions
		_start = findInstAt(Address(_file->entry()));
		return file;
	}

	otawa::Inst *decode(Address addr) {
		riscv_inst_t *inst = riscv_decode(_decoder, addr.offset());
		Inst::kind_t kind = 0;
		otawa::Inst *result = 0;
		// TODO
		kind = riscv_kind(inst);
		if(kind & Inst::IS_CONTROL)
			result = new BranchInst(*this, kind, addr);
		else
			result = new Inst(*this, kind, addr);
		ASSERT(result);
		riscv_free_inst(inst);
		return result;
	}

	gel::File *file(void) const { return _file; }
	riscv_memory_t *memory(void) const { return _memory; }
	inline riscv_decoder_t *decoder() const { return _decoder; }
	inline void *platform(void) const { return _platform; }

	Option<Pair<cstring, int> > getSourceLine(Address addr) override {
		setup_debug();
		if (map == nullptr)
			return none;
		auto line = map->lineAt(addr.offset());
		if(line == nullptr)
			return none;
		else
			return some(pair(
				line->file()->path().toString().toCString(),
				line->line()));
	}

	void getAddresses(
		cstring file,
		int line, Vector<Pair<Address, Address> >& addresses
	) override {
		setup_debug();
		addresses.clear();
		if(map == nullptr)
			return;
		auto source = map->files().get(file, nullptr);
		if(source != nullptr) {
			Vector<Pair<gel::address_t, gel::address_t> > rs;
			source->find(line, rs);
			for(auto p: rs)
				addresses.add(pair(Address(p.fst), Address(p.snd)));
		}
	}

	void get(Address at, t::int8& val) override
		{ val = riscv_mem_read8(_memory, at.offset()); }
	void get(Address at, t::uint8& val) override
		{ val = riscv_mem_read8(_memory, at.offset()); }
	void get(Address at, t::int16& val) override
		{ val = riscv_mem_read16(_memory, at.offset()); }
	void get(Address at, t::uint16& val) override
		{ val = riscv_mem_read16(_memory, at.offset()); }
	void get(Address at, t::int32& val) override
		{ val = riscv_mem_read32(_memory, at.offset()); }
	void get(Address at, t::uint32& val) override
		{ val = riscv_mem_read32(_memory, at.offset()); }
	void get(Address at, t::int64& val) override
		{ val = riscv_mem_read64(_memory, at.offset()); }
	void get(Address at, t::uint64& val) override
		{ val = riscv_mem_read64(_memory, at.offset()); }
	void get(Address at, Address& val) override
		{ val = riscv_mem_read32(_memory, at.offset()); }
	void get(Address at, string& str)  override {
		Address base = at;
		while(!riscv_mem_read8(_memory, at.offset()))
			at = at + 1;
		int len = at - base;
		char buf[len];
		get(base, buf, len);
		str = String(buf, len);
	}
	void get(Address at, char *buf, int size) override
		{ riscv_mem_read(_memory, at.offset(), buf, size); }

	void dump(io::Output& out, Address addr) {
		char out_buffer[200];
		riscv_inst_t *inst = riscv_decode(_decoder, addr.offset());
		riscv_disasm(out_buffer, inst);
		riscv_free_inst(inst);
		out << out_buffer;
	}

	Address decodeTarget(Address a) {
		riscv_inst_t *inst = riscv_decode(_decoder, a.offset());
		Address target_addr = riscv_target(inst);
		riscv_free_inst(inst);
		return target_addr;
	}

	delayed_t decodeDelayed(Address a) {
		riscv_inst_t *inst= riscv_decode(_decoder, a.offset());
		delayed_t d = delayed_t(riscv_delayed(inst));
		riscv_free_inst(inst);
		return d;
	}

	void decodeReadRegSet(Address a, RegSet &set) {
		riscv_inst_t *inst= riscv_decode(_decoder, a.offset());
		riscv_read(inst, set);
		riscv_free_inst(inst);
	}

	// writeRegSet from Inst
	void decodeWriteRegSet(Address a, RegSet &set) {
		riscv_inst_t *inst= riscv_decode(_decoder, a.offset());
		riscv_write(inst, set);
		riscv_free_inst(inst);
	}

private:

	void setup_debug(void) {
		ASSERT(_file != nullptr);
		if(init)
			return;
		init = true;
		map = _file->debugLines();
	}

	otawa::Inst *_start;
	hard::Platform *oplatform;
	riscv_platform_t *_platform;
	riscv_memory_t *_memory;
	riscv_decoder_t *_decoder;
	gel::DebugLine *map;
	gel::File *_file;
	int argc;
	AllocArray<cstring> argv, envp;
	bool no_stack;
	bool init;
};


// decode from Segment
otawa::Inst *Segment::decode(address_t address) {
	return proc.decode(address);
}

// dump from Inst
void Inst::dump(io::Output& out) {
	proc.dump(out, _addr);
}

// semInsts from Inst
void Inst::semInsts(sem::Block &block) {
	// TO DO
}

// target from BranchInst
otawa::Inst *BranchInst::target(void) {
	if (!isTargetDone) {
		isTargetDone = true;
		if(!isIndirect()) {
			riscv_address_t a = proc.decodeTarget(_addr).offset();
			_target = process().findInstAt(a);
		}
	}
	return _target;
}


const Array<hard::Register *>& Inst::readRegs(void) {
	// a bit ugly but this method is deprecated
	static Vector<AllocArray<hard::Register *> *> tabs;
	RegSet set;
	readRegSet(set);
	if(tabs.length() < set.count() + 1) {
		int osize = tabs.length();
		tabs.setLength(set.count() + 1);
		for(int i = osize; i < tabs.length(); i++)
			tabs[i] = nullptr;
	}
	if(tabs[set.count()] == nullptr)
		tabs[set.count()] = new AllocArray<hard::Register *>(set.count());
	AllocArray<hard::Register *>& t = *tabs[set.count()];
	for(int i = 0; i < set.count(); i++)
		t[i] = proc.platform()->findReg(set[i]);
	return t;
}

const Array<hard::Register *>& Inst::writtenRegs(void) {
	// a bit ugly but this method is deprecated
	static Vector<AllocArray<hard::Register *> *> tabs;
	RegSet set;
	writeRegSet(set);
	if(tabs.length() < set.count() + 1) {
		int osize = tabs.length();
		tabs.setLength(set.count() + 1);
		for(int i = osize; i < tabs.length(); i++)
			tabs[i] = nullptr;
	}
	if(tabs[set.count()] == nullptr)
		tabs[set.count()] = new AllocArray<hard::Register *>(set.count());
	AllocArray<hard::Register *>& t = *tabs[set.count()];
	for(int i = 0; i < set.count(); i++)
		t[i] = proc.platform()->findReg(set[i]);
	return t;
}

// delayType for BranchInst
delayed_t BranchInst::delayType(void) {
	return proc.decodeDelayed(_addr);
}

// readRegSet from Inst
void Inst::readRegSet (RegSet &set) {
	proc.decodeReadRegSet(_addr, set);

}

// writeRegSet from Inst
void Inst::writeRegSet(RegSet &set) {
	proc.decodeWriteRegSet(_addr, set);
}


// Loader definition
class Loader: public otawa::Loader {
public:
	Loader(void): otawa::Loader(make("riscv", OTAWA_LOADER_VERSION).version(ISA_VERSION).alias("elf_243")) {
	}

	virtual CString getName(void) const { return "riscv"; }

	virtual otawa::Process *load(Manager *man, CString path, const PropList& props) {
		otawa::Process *proc = create(man, props);
		if (!proc->loadProgram(path)) {
			delete proc;
			return 0;
		}
		else {

			return proc;
		}
	}

	virtual otawa::Process *create(Manager *man, const PropList& props)
		{ return new Process(man, new Platform(props), props); }
};


// plugin definition
class Plugin: public otawa::ProcessorPlugin {
public:
	Plugin(void): otawa::ProcessorPlugin("otawa/riscv", Version(ISA_VERSION), OTAWA_PROC_VERSION) {
	}
};


} }	// otawa::riscv

otawa::riscv::Loader otawa_riscv_loader;
ELM_PLUGIN(otawa_riscv_loader, OTAWA_LOADER_HOOK);
otawa::riscv::Plugin riscv_plugin;
ELM_PLUGIN(riscv_plugin, OTAWA_PROC_HOOK);


